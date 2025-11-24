from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from config.settings import settings
from io.redis_frames import RedisFrameClient
from models.action_classifier import ActionClassifier, QwenAction
from models.detection import Detection, PersonDetector

logger = logging.getLogger(__name__)


@dataclass
class PersonActionResult:
    frame_index: int
    person_index: int
    box_xyxy: tuple[int, int, int, int]
    grounding_score: float
    crop_path: Optional[str]
    qwen_action: QwenAction


class PosePipeline:
    def __init__(self, detector: PersonDetector, classifier: ActionClassifier) -> None:
        self.detector = detector
        self.classifier = classifier
        self.redis_client = RedisFrameClient()
        self._stop_flags: Dict[str, bool] = {}

    def request_stop(self, stream_key: Optional[str] = None) -> None:
        if stream_key is None:
            for k in list(self._stop_flags.keys()):
                self._stop_flags[k] = True
        else:
            self._stop_flags[stream_key] = True

    async def process_stream_async(
        self,
        stream_key: str,
        output_dir: Optional[str] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
        manage_classifier: bool = True,
    ) -> None:
        if manage_classifier:
            await self.classifier.start(settings.vllm.max_workers)

        output_root = Path(output_dir) if output_dir else None
        if output_root:
            output_root.mkdir(parents=True, exist_ok=True)
            if (output_root / "crops").exists() is False:
                (output_root / "crops").mkdir(parents=True, exist_ok=True)

        stop_flag = False
        self._stop_flags[stream_key] = False
        frame_idx = 0
        while not stop_flag:
            msg_id, meta, jpeg = await self.redis_client.get_frame(
                stream_key, timeout_ms=int(settings.pipeline.timeout_sec * 1000)
            )
            stop_flag = self._stop_flags.get(stream_key, False)
            if not msg_id or jpeg is None:
                await asyncio.sleep(0.01)
                continue

            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_idx += 1
            if frame_idx % settings.pipeline.frame_interval != 0:
                continue

            t_frame_start = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections: List[Detection] = self.detector.detect(Image.fromarray(frame_rgb))
            if not detections:
                continue

            per_frame_results: List[PersonActionResult] = []
            jobs: List[asyncio.Future[QwenAction]] = []
            expanded_boxes: List[tuple[int, int, int, int]] = []

            for pid, det in enumerate(detections):
                x1, y1, x2, y2 = PersonDetector.expand_box(
                    det.box_xyxy,
                    settings.thresholds.box_expand_ratio,
                    frame.shape[1],
                    frame.shape[0],
                )
                crop = frame[y1:y2, x1:x2].copy()
                crop, _, _ = PersonDetector.ensure_min_short_side(
                    crop, settings.thresholds.crop_min_short_side
                )
                crop_path = None
                if output_root:
                    crop_path = str(output_root / "crops" / f"frame_{frame_idx:06d}_p{pid:02d}.jpg")
                    cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                meta_payload = {
                    "frame_index": frame_idx,
                    "person_index": pid,
                    "msg_id": msg_id,
                }
                expanded_boxes.append((x1, y1, x2, y2))
                jobs.append(self.classifier.submit(crop, meta_payload))

            actions = await asyncio.gather(*jobs)
            for pid, (det, action, box) in enumerate(zip(detections, actions, expanded_boxes)):
                result = PersonActionResult(
                    frame_index=frame_idx,
                    person_index=pid,
                    box_xyxy=box,
                    grounding_score=det.score,
                    crop_path=None,
                    qwen_action=action,
                )
                per_frame_results.append(result)
                logger.info(
                    json.dumps(
                        {
                            "event": "person",
                            "frame_idx": frame_idx,
                            "person_index": pid,
                            "class_name": action.class_name,
                            "confidence": action.confidence,
                            "http_ms": action.timings.get("http_ms"),
                            "total_ms": action.timings.get("total_ms"),
                        }
                    )
                )

            t_frame_end = time.perf_counter()
            batch_ms = (t_frame_end - t_frame_start) * 1000.0
            logger.info(
                json.dumps(
                    {
                        "event": "frame",
                        "frame_idx": frame_idx,
                        "num_persons": len(detections),
                        "batch_ms": batch_ms,
                        "vllm_workers": settings.vllm.max_workers,
                    }
                )
            )

            if on_result:
                for res, box in zip(per_frame_results, expanded_boxes):
                    payload = {
                        "expanded_box": box,
                        "preds": [
                            {
                                "class_name": res.qwen_action.class_name,
                                "confidence": res.qwen_action.confidence,
                                "rationale": res.qwen_action.rationale,
                                "cls_id": res.qwen_action.class_id,
                                "version": settings.vllm.model,
                            }
                        ],
                        "frame_index": res.frame_index,
                        "person_index": res.person_index,
                    }
                    on_result(payload)

            if settings.pipeline.max_frames and frame_idx >= settings.pipeline.max_frames:
                break

        if manage_classifier:
            await self.classifier.stop()

    def process_stream(
        self,
        stream_key: str,
        output_dir: Optional[str] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
        manage_classifier: bool = True,
    ) -> None:
        asyncio.run(
            self.process_stream_async(
                stream_key=stream_key,
                output_dir=output_dir,
                on_result=on_result,
                manage_classifier=manage_classifier,
            )
        )
