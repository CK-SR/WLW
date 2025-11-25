from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import asyncio


import cv2
import numpy as np
import torch
from PIL import Image
import redis
from configs.settings import settings as config_settings
from camera_check_fastapi.src.performance import PERFORMANCE_MONITOR
# from transformers import (
#     AutoModelForZeroShotObjectDetection,
#     AutoProcessor,
    
# )

from ultralytics import YOLO


import io
import base64
import mimetypes
import requests

import concurrent.futures


# from modelscope import Qwen3VLForConditionalGeneration

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_ACTION_CLASSES: Dict[int, str] = {
    0: "smoking",
    1: "calling",
    2: "climb",
    3: "jump",
    4: "run",
    5: "takephoto",
    6: "walk",
    7: "carry",
}


def _parse_iso_to_epoch(ts: Optional[str]) -> Optional[float]:
    """Parse ISO-8601 or numeric timestamp string to epoch seconds."""

    if ts is None:
        return None
    try:
        return float(ts)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except Exception:
        return None


MONITORING_ENABLED = bool(config_settings.monitoring.enabled)
if MONITORING_ENABLED:
    PERFORMANCE_MONITOR.configure(
        history_size=int(config_settings.monitoring.history_size),
        frame_history=int(config_settings.monitoring.frame_history),
    )
else:
    PERFORMANCE_MONITOR.reset()


@dataclass
class GroundingResult:
    """Container for a single Grounding DINO detection."""

    box_xyxy: Tuple[float, float, float, float]
    score: float
    label: str


@dataclass
class QwenAction:
    """Structured output from Qwen action classification."""

    class_id: Optional[int]
    class_name: Optional[str]
    confidence: Optional[float]
    rationale: Optional[str]
    raw_response: str
    # 新增：记录本次 vLLM 调用相关的耗时信息（毫秒）和一些元信息
    timings: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonActionResult:
    """Aggregate result for a detected person."""

    frame_index: int
    person_index: int
    box_xyxy: Tuple[int, int, int, int]
    grounding_score: float
    crop_path: Optional[str]
    qwen_action: QwenAction

EXECUTOR_IO = concurrent.futures.ThreadPoolExecutor(max_workers=64)
r = redis.Redis(
    host="192.168.130.14",
    port=6379,
    password=None,
    db=0,
    decode_responses=False,  # 保持二进制，和你当前用法一致
)


def _read_one_from_redis(stream_key: str, block_ms: int = 1000):
    """
    注意：这里的参数现在是完整的 Redis Stream key，例如：
        "frames:rtsp://192.168.1.10:8554/cam01"
    """
    return r.xread({f"frames:{stream_key}": "$"}, count=1, block=block_ms) or []


async def get_one_frame(stream_key: str, timeout_sec: float = 1.0):
    """
    从指定的 Redis Stream key 读取一条记录，返回 (msg_id, meta, jpeg, redis_ms)。

    :param stream_key: 完整的 Redis key，例如 "frames:rtsp://192.168.1.10:8554/cam01"
    """
    loop = asyncio.get_running_loop()
    deadline = time.time() + timeout_sec
    start_monotonic = time.monotonic()
    while time.time() < deadline:
        streams = await loop.run_in_executor(
            EXECUTOR_IO, _read_one_from_redis, stream_key, int(timeout_sec * 1000)
        )
        if streams:
            elapsed_ms = (time.monotonic() - start_monotonic) * 1000.0
            print(f"[redis] got frame for {stream_key}")
            for _stream, messages in streams:
                for _id, fields in messages:
                    meta = json.loads(fields[b"meta"].decode("utf-8"))
                    jpeg = fields[b"jpeg"]
                    msg_id = _id.decode() if isinstance(_id, bytes) else str(_id)
                    if MONITORING_ENABLED:
                        PERFORMANCE_MONITOR.record_stage(stream_key, "redis_fetch", elapsed_ms)
                    return msg_id, meta, jpeg, elapsed_ms
        else:
            print(f"[redis] no frame for {stream_key}, retry...")
        await asyncio.sleep(0.01)
    return None, None, None, None

def _load_class_names(data_config: Optional[str]) -> Dict[int, str]:
    if not data_config:
        return {}

    yaml_path = Path(data_config)
    if not yaml_path.exists():
        return {}

    try:
        text = yaml_path.read_text(encoding="utf-8")
    except Exception:
        return {}

    if yaml is not None:
        try:
            payload = yaml.safe_load(text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            names = payload.get("names")
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, list):
                return {idx: str(name) for idx, name in enumerate(names)}

    mapping: Dict[int, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        left = left.strip()
        right = right.strip().strip("'\"")
        if not left.isdigit():
            continue
        mapping[int(left)] = right
    return mapping




def _expand_box(
    box_xyxy: Iterable[float],
    expand_ratio: float,
    img_width: int,
    img_height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box_xyxy)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    dx = width * expand_ratio
    dy = height * expand_ratio

    nx1 = max(0, int(round(x1 - dx)))
    ny1 = max(0, int(round(y1 - dy)))
    nx2 = min(img_width - 1, int(round(x2 + dx)))
    ny2 = min(img_height - 1, int(round(y2 + dy)))

    if nx2 <= nx1:
        nx2 = min(img_width - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(img_height - 1, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _ensure_min_short_side(
    crop: np.ndarray,
    target_short_side: int,
) -> Tuple[np.ndarray, float, float]:
    height, width = crop.shape[:2]
    if height == 0 or width == 0:
        return crop, 1.0, 1.0

    short_side = min(height, width)
    if short_side >= target_short_side:
        return crop, 1.0, 1.0

    scale = target_short_side / short_side
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    scale_x = new_width / width
    scale_y = new_height / height
    return resized, scale_x, scale_y


def _create_video_writer(
    video_path: Path,
    fps: float,
    frame_size: Tuple[int, int],
) -> cv2.VideoWriter:
    codecs = ["mp4v", "avc1", "H264", "XVID"]
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
        if writer.isOpened():
            return writer
    raise RuntimeError("Unable to create VideoWriter for annotated output mp4")


def _load_prompt_text(path: Optional[str]) -> str:
    if not path:
        return ""
    prompt_path = Path(path)
    if not prompt_path.exists():
        return ""
    try:
        raw_text = prompt_path.read_text(encoding="utf-8")
        print("get prompt success")
    except Exception:
        return ""

    if yaml is not None:
        try:
            payload = yaml.safe_load(raw_text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            if isinstance(payload.get("content"), str):
                return payload["content"].strip()
            if isinstance(payload.get("prompt"), str):
                return payload["prompt"].strip()
    return raw_text.strip()


def _extract_json_fragment(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.startswith("```")]
        stripped = "\n".join(lines).strip()

    try:
        start = stripped.index("{")
        end = stripped.rindex("}") + 1
    except ValueError:
        return None
    return stripped[start:end]


def _img_ndarray_to_data_url(
    bgr: np.ndarray,
    fmt: str = "jpeg",
    quality: int = 95,
) -> Tuple[str, float, int]:
    """
    OpenCV 的 BGR ndarray -> data:URL（vLLM OpenAI chat 支持 image_url.data）
    返回: (data_url, encode_ms, jpeg_bytes)
    """
    t0 = time.perf_counter()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image  # 局部导入以避免循环依赖
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    save_params = {}
    if fmt.lower() == "jpeg":
        save_params["quality"] = quality
        mime = "image/jpeg"
    elif fmt.lower() == "png":
        mime = "image/png"
    else:
        mime = mimetypes.types_map.get("." + fmt.lower(), "image/jpeg")
    pil_img.save(buf, format=fmt.upper(), **save_params)
    jpeg_bytes = buf.getvalue()
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    t1 = time.perf_counter()
    encode_ms = (t1 - t0) * 1000.0

    return data_url, encode_ms, len(jpeg_bytes)


def _call_vllm_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_tokens: int = 256,
    timeout: float = 60.0,
) -> str:
    """
    调用 vLLM 的 OpenAI 兼容 /v1/chat/completions，返回 message.content（字符串）。
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",  # vLLM 默认不校验
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    # vLLM 可能返回富内容 list，这里抽取 text
    if isinstance(content, list):
        parts = []
        for it in content:
            if isinstance(it, dict) and isinstance(it.get("text"), str):
                parts.append(it["text"])
        content = "\n".join(parts)
    return str(content or "")

class QwenPersonActionPipeline:
    """Pipeline that combines Grounding DINO detection with Qwen action reasoning."""

    def __init__(
        self,
        grounding_model_id: str = "~/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny",
        text_query: str = "person. ",
        grounding_threshold: float = 0.6,
        text_threshold: float = 0.25,
        qwen_model_dir: str = "/home/cs/leiheao/dataengine/Qwenvl",
        system_prompt_path: Optional[str] = "/home/cs/leiheao/dataengine/prompt/system.yaml",
        user_prompt_path: Optional[str] = "/home/cs/leiheao/dataengine/prompt/user.yaml",
        data_config: Optional[str] = "/home/cs/leiheao/dataengine/data_10_28_datav6.yaml",
        device: Optional[str] = "cuda",
        qwen_device: Optional[str] = "cuda",
        qwen_max_new_tokens: int = 128,
        qwen_temperature: float = 0.0,
        qwen_top_p: float = 0.9,
        use_flash_attention: bool = True,
        qwen_model_name: str = "Qwen2.5-VL",
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._text_query = text_query
        self._grounding_threshold = grounding_threshold
        self._text_threshold = text_threshold

        self._system_prompt = _load_prompt_text(system_prompt_path)
        self._user_prompt = _load_prompt_text(user_prompt_path)


        # self._processor = AutoProcessor.from_pretrained(grounding_model_id)
        # self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        #     grounding_model_id
        # ).to(self._device)

        self._grounding_model = YOLO("/workspace/models/yolo11m.pt").to(self._device)

        self._class_names = _load_class_names(data_config) or DEFAULT_ACTION_CLASSES
        self._class_name_to_id = {
            name.lower(): idx for idx, name in self._class_names.items()
        }

        self._qwen_device = qwen_device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if "cuda" in str(self._qwen_device) and torch.cuda.is_available() else torch.float32
        #self._qwen_processor = AutoProcessor.from_pretrained(qwen_model_dir)
        
        self._qwen_max_new_tokens = int(max(1, qwen_max_new_tokens))
        self._qwen_temperature = max(0.0, float(qwen_temperature))
        self._qwen_top_p = float(qwen_top_p)
        # === 下面这些是保留原有推理超参（传给 vLLM 的 sampling 配置） ===
        self._qwen_max_new_tokens = int(max(1, qwen_max_new_tokens))
        self._qwen_temperature = max(0.0, float(qwen_temperature))
        self._qwen_top_p = float(qwen_top_p)

        # === vLLM 连接信息：可被环境变量覆盖 ===
        # 你的 vLLM 在 http://192.168.130.14:8010/v1 ，模型名按 vLLM --model 的名字
        self._vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://192.168.130.14:8010/v1")
        self._vllm_model = os.environ.get("VLLM_MODEL", "/model/Qwen3-VL-8B-Instruct")

        # 保留 prompt 路径，供分类时拼接 system/user 指令
        self._system_prompt_path = system_prompt_path
        self._user_prompt_path = user_prompt_path
        # === vLLM 并发执行线程池 ===
        # 环境变量 VLLM_MAX_WORKERS 可覆盖默认值 8
        self._vllm_workers = int(os.environ.get("VLLM_MAX_WORKERS", "8"))
        self._vllm_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._vllm_workers
        )

        # ==== 新增：每路流一个 stop_flag，用于 /stop 控制 ====
        self._stop_flags: Dict[str, bool] = {}

    # def _detect_with_grounding(self, frame_pil: Image.Image) -> List[GroundingResult]:
    #     inputs = self._processor(
    #         images=frame_pil, text=self._text_query, return_tensors="pt"
    #     ).to(self._device)

    #     with torch.inference_mode():
    #         outputs = self._grounding_model(**inputs)

    #     post_processed = self._processor.post_process_grounded_object_detection(
    #         outputs,
    #         input_ids=inputs.input_ids,
    #         threshold=self._grounding_threshold,
    #         text_threshold=self._text_threshold,
    #         target_sizes=[frame_pil.size[::-1]],
    #     )

    #     results: List[GroundingResult] = []
    #     if not post_processed:
    #         return results

    #     data = post_processed[0]
    #     scores = data.get("scores", [])
    #     labels = data.get("labels", [])
    #     boxes = data.get("boxes", [])
    #     for score, label, box in zip(scores, labels, boxes):
    #         label_text = str(label).lower()
    #         if "person" not in label_text:
    #             continue
    #         results.append(
    #             GroundingResult(box_xyxy=tuple(box.tolist()), score=float(score), label=label)
    #         )
    #     return results

    def _detect_with_grounding(self, frame_pil: Image.Image) -> List[GroundingResult]:
        results = self._grounding_model(frame_pil, conf=self._grounding_threshold, classes=[0])  # 0=COCO person
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                score = float(box.conf[0].cpu())
                detections.append(GroundingResult(box_xyxy=(x1, y1, x2, y2), score=score, label="person"))
        return detections

    def _build_messages(self, image: Image.Image) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self._system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._system_prompt}],
                }
            )
        user_content = [{"type": "image", "image": image}]
        if self._user_prompt:
            user_content.append({"type": "text", "text": self._user_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def request_stop(self, stream_key: Optional[str] = None) -> None:
        """
        外部调用，用于通知某一路或全部流停止处理。
        """
        if stream_key is None:
            # 停止所有
            for k in list(self._stop_flags.keys()):
                self._stop_flags[k] = True
        else:
            self._stop_flags[stream_key] = True

    def _classify_action_with_qwen(self, crop: np.ndarray) -> QwenAction:
        # === (0) 总计时开始 ===
        t_start = time.perf_counter()

        # === (1) BGR -> data:URL（vLLM 接收 image_url），记录编码耗时 & 体积 ===
        data_url, encode_ms, jpeg_bytes = _img_ndarray_to_data_url(
            crop, fmt="jpeg", quality=95
        )

        # === (2) 组装 system / user（用你已有的 system.yaml / user.yaml 文本），记录构造消息耗时 ===
        t_build0 = time.perf_counter()

        system_text = self._system_prompt or ""
        # 追加一个“只返回 JSON”约束，增强解析稳定性
        system_text = (system_text + "\n\n"
                    "You MUST answer with STRICT JSON only, no extra text or markdown fences. "
                    'Keys: {"class_id": int, "class_name": string, "confidence": float (0..1), "rationale": string}.').strip()

        user_text = (self._user_prompt or "").strip()
        if user_text:
            user_text = user_text + "\n\nAnalyze this single person crop and output JSON only."
        else:
            user_text = "Analyze this single person crop and output JSON only."

        messages: List[Dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })

        t_build1 = time.perf_counter()
        build_msg_ms = (t_build1 - t_build0) * 1000.0

        # === (3) 调 vLLM OpenAI 接口，记录 HTTP + 服务端推理 + r.json() 的总耗时 ===
        t_http0 = time.perf_counter()
        http_error: Optional[str] = None
        try:
            response_text = _call_vllm_chat(
                base_url=self._vllm_base_url,
                model=self._vllm_model,
                messages=messages,
                temperature=self._qwen_temperature,
                top_p=self._qwen_top_p,
                max_tokens=self._qwen_max_new_tokens,
                timeout=60.0,
            )
        except Exception as e:
            http_error = str(e)
            # 出错时也返回一个可解析的 JSON
            response_text = json.dumps({
                "class_id": None,
                "class_name": None,
                "confidence": None,
                "rationale": f"vLLM request failed: {e}",
            }, ensure_ascii=False)
        t_http1 = time.perf_counter()
        http_ms = (t_http1 - t_http0) * 1000.0

        # === (4) 解析模型返回的 JSON，记录解析耗时 ===
        t_parse0 = time.perf_counter()
        json_fragment = _extract_json_fragment(response_text) or response_text
        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(json_fragment)
        except Exception:
            parsed = {}
        t_parse1 = time.perf_counter()
        parse_json_ms = (t_parse1 - t_parse0) * 1000.0

        # === (5) 提取字段（保持你原有逻辑） ===
        class_id: Optional[int] = None
        class_name: Optional[str] = None
        confidence: Optional[float] = None
        rationale: Optional[str] = None

        if isinstance(parsed.get("class_id"), int):
            class_id = int(parsed["class_id"])
        elif isinstance(parsed.get("class_id"), str) and parsed["class_id"].isdigit():
            class_id = int(parsed["class_id"])

        if isinstance(parsed.get("class_name"), str):
            class_name = parsed["class_name"].strip()

        # name / id 互补
        if class_name and class_id is None:
            class_id = self._class_name_to_id.get(class_name.lower())
        if class_id is not None and not class_name:
            class_name = self._class_names.get(class_id)

        if isinstance(parsed.get("confidence"), (float, int, str)):
            try:
                confidence = float(parsed["confidence"])
                confidence = float(max(0.0, min(1.0, confidence)))
            except Exception:
                confidence = None

        if isinstance(parsed.get("rationale"), str):
            rationale = parsed["rationale"].strip()

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0

        # === (6) 本次调用的耗时和元信息记录到 QwenAction 里 ===
        timings = {
            "encode_ms": encode_ms,
            "build_msg_ms": build_msg_ms,
            "http_ms": http_ms,
            "parse_json_ms": parse_json_ms,
            "total_ms": total_ms,
        }
        if http_error:
            timings["http_error"] = 1.0

        h, w = crop.shape[:2]
        meta = {
            "jpeg_bytes": jpeg_bytes,
            "crop_wh": (w, h),
        }
        if http_error:
            meta["http_error_msg"] = http_error

        # 可以临时打印一下（调试阶段用，之后可以注释掉）
        print(
            f"[vLLM] encode={encode_ms:.2f}ms, "
            f"build_msg={build_msg_ms:.2f}ms, http={http_ms:.2f}ms, "
            f"parse_json={parse_json_ms:.2f}ms, total={total_ms:.2f}ms, "
            f"jpeg={jpeg_bytes/1024:.1f}KB"
        )

        return QwenAction(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            rationale=rationale,
            raw_response=response_text,
            timings=timings,
            meta=meta,
        )

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        frame_interval: int = 1,
        box_expand_ratio: float = 0.4,
        crop_min_short_side: int = 448,
        save_crops: bool = True,
        save_anno: bool = True,
        save_video: bool = True,
        top_k: int = 1,
        annotated_video_name: str = "annotated",
    ) -> List[PersonActionResult]:
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        save_root = Path(output_dir) if output_dir else None
        if save_root:
            save_root.mkdir(parents=True, exist_ok=True)
        if save_root and save_crops:
            (save_root / "crops").mkdir(parents=True, exist_ok=True)
        if save_root and save_anno:
            (save_root / "annotated_frames").mkdir(parents=True, exist_ok=True)

        video_output_path: Optional[Path] = None
        writer: Optional[cv2.VideoWriter] = None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if save_root and save_video:
            video_output_path = (save_root / annotated_video_name).with_suffix(".mp4")
            video_output_path.parent.mkdir(parents=True, exist_ok=True)

        results: List[PersonActionResult] = []
        frame_idx = 0
        top_k = max(1, int(top_k))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        progress = None
        if tqdm is not None:
            progress = tqdm(
                total=total_frames if total_frames > 0 else None,
                desc="Processing video frames",
                unit="frame",
            )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if progress is not None:
                    progress.update(1)

                frame_idx += 1
                if frame_idx % frame_interval != 0:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                img_width, img_height = frame_pil.size

                detections = self._detect_with_grounding(frame_pil)
                print(f"Frame {frame_idx}: Detected {len(detections)} persons")
                print(f"detections: {detections}")
                if not detections:
                    continue

                per_frame_results: List[Dict[str, Any]] = []

                # === (A) 准备每个人的裁剪和 meta 信息，同时提交 vLLM 任务 ===
                person_jobs: List[Dict[str, Any]] = []

                t_batch_start = time.perf_counter()

                for person_id, detection in enumerate(detections):
                    x1, y1, x2, y2 = _expand_box(
                        detection.box_xyxy, box_expand_ratio, img_width, img_height
                    )
                    crop = frame[y1:y2, x1:x2].copy()
                    crop, _, _ = _ensure_min_short_side(
                        crop, crop_min_short_side
                    )

                    crop_path = None
                    if save_root and save_crops:
                        crop_name = f"frame_{frame_idx:06d}_person_{person_id:02d}.jpg"
                        crop_path = str(save_root / "crops" / crop_name)
                        cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    # 用线程池并发调用 vLLM
                    future = self._vllm_executor.submit(
                        self._classify_action_with_qwen, crop
                    )

                    person_jobs.append(
                        {
                            "person_id": person_id,
                            "detection": detection,
                            "box": (x1, y1, x2, y2),
                            "crop_path": crop_path,
                            "future": future,
                        }
                    )

                # === (B) 等待所有 vLLM 结果返回，并填充 per_frame_results / results ===
                for job in person_jobs:
                    person_id = job["person_id"]
                    detection = job["detection"]
                    x1, y1, x2, y2 = job["box"]
                    crop_path = job["crop_path"]
                    future = job["future"]

                    # .result() 会在必要时阻塞，但多个请求是在并发进行的
                    qwen_action: QwenAction = future.result()

                    per_frame_results.append(
                        {
                            "expanded_box": (x1, y1, x2, y2),
                            "preds": [
                                {
                                    "class_name": qwen_action.class_name,
                                    "confidence": qwen_action.confidence,
                                    "rationale": qwen_action.rationale,
                                }
                            ],
                        }
                    )

                    results.append(
                        PersonActionResult(
                            frame_index=frame_idx,
                            person_index=person_id,
                            box_xyxy=(x1, y1, x2, y2),
                            grounding_score=detection.score,
                            crop_path=crop_path,
                            qwen_action=qwen_action,
                        )
                    )

                t_batch_end = time.perf_counter()
                batch_ms = (t_batch_end - t_batch_start) * 1000.0

                # 简单打个 log 看看这一帧 vLLM 总耗时（并发后是“批次耗时”）
                print(
                    f"[Frame {frame_idx}] vLLM batch for {len(person_jobs)} persons "
                    f"took {batch_ms:.2f} ms (workers={self._vllm_workers})"
                )

                has_detections = bool(per_frame_results)
                need_video_frame = video_output_path is not None
                need_image_frame = bool(save_root and save_anno and has_detections)
                if need_video_frame or need_image_frame:
                    annotated = frame.copy()
                    if has_detections:
                        for person_data in per_frame_results:
                            x1, y1, x2, y2 = person_data["expanded_box"]
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 200, 80), 2)
                            preds = person_data.get("preds", [])
                            for rank, pred in enumerate(preds[:top_k]):
                                label_text = pred.get("class_name") or "unknown"
                                conf = pred.get("confidence")
                                if conf is not None:
                                    label_text += f" {conf * 100:.1f}%"
                                y_anchor = max(15, y1 - 10 - rank * 18)
                                cv2.putText(
                                    annotated,
                                    label_text,
                                    (x1, y_anchor),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (50, 220, 50),
                                    2,
                                    lineType=cv2.LINE_AA,
                                )

                    if need_video_frame:
                        if writer is None and video_output_path is not None:
                            frame_size = (annotated.shape[1], annotated.shape[0])
                            writer = _create_video_writer(video_output_path, fps, frame_size)
                        if writer is not None:
                            writer.write(annotated)

                    if need_image_frame and save_root is not None:
                        annotated_name = f"frame_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(save_root / "annotated_frames" / annotated_name), annotated)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if progress is not None:
                progress.close()

        return results

    def process_stream(
        self,
        stream_key: str,
        output_dir: Optional[str] = None,
        frame_interval: int = 1,
        box_expand_ratio: float = 0.4,
        crop_min_short_side: int = 448,
        save_crops: bool = True,
        save_anno: bool = True,
        save_video: bool = True,
        top_k: int = 1,
        annotated_video_name: str = "annotated",
        max_frames: Optional[int] = None,    # 最多处理多少帧，为 None 就一直跑
        timeout_sec: float = 1.0,            # 每次从 redis 等待一帧的超时时间
        on_result: Optional[callable] = None,  # 新增：每帧检测结果的回调，用于 WebSocket 推送
    ) -> List[PersonActionResult]:
        """
        从 Redis 中的帧流（frames:{stream_name}）读取视频帧进行处理。
        """
        save_root = Path(output_dir) if output_dir else None
        if save_root:
            save_root.mkdir(parents=True, exist_ok=True)
        if save_root and save_crops:
            (save_root / "crops").mkdir(parents=True, exist_ok=True)
        if save_root and save_anno:
            (save_root / "annotated_frames").mkdir(parents=True, exist_ok=True)

        video_output_path: Optional[Path] = None
        writer: Optional[cv2.VideoWriter] = None
        # 流一般没 fps，这里给个默认值，或者从 meta 里取
        fps = 10.0

        results: List[PersonActionResult] = []
        frame_idx = 0
        top_k = max(1, int(top_k))

        # 没有总帧数，所以 tqdm 用 total=None
        progress = None
        if tqdm is not None:
            progress = tqdm(
                total=None,
                desc=f"Processing stream {stream_key}",
                unit="frame",
            )

        # 为这一条流初始化 stop_flag
        self._stop_flags.setdefault(stream_key, False)

        try:
            while True:
                # === 检查是否收到停止信号 ===
                if self._stop_flags.get(stream_key, False):
                    print(f"[stream {stream_key}] received stop signal, stop.")
                    break

                if max_frames is not None and frame_idx >= max_frames:
                    print(f"[stream {stream_key}] reached max_frames={max_frames}, stop.")
                    break

                # === 使用新版 get_one_frame，从 Redis 读一帧 ===
                fetch_start = time.perf_counter()
                msg_id, meta, jpeg, redis_fetch_ms = asyncio.run(
                    get_one_frame(stream_key, timeout_sec=timeout_sec)
                )
                fetch_end = time.perf_counter()
                if redis_fetch_ms is None:
                    redis_fetch_ms = (fetch_end - fetch_start) * 1000.0
                if jpeg is None:
                    print(f"[stream {stream_key}] timeout {timeout_sec}s, no frame, stop.")
                    break

                decode_start = time.perf_counter()

                np_buf = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                decode_end = time.perf_counter()
                decode_ms = (decode_end - decode_start) * 1000.0
                if frame is None:
                    print(f"[stream {stream_key}] decode failed, skip this frame.")
                    if MONITORING_ENABLED:
                        ts1_epoch = _parse_iso_to_epoch(meta.get("ts1")) if meta else None
                        ts2_epoch = _parse_iso_to_epoch(meta.get("ts2")) if meta else None
                        ts3_epoch = time.time()
                        cap_ms = (ts2_epoch - ts1_epoch) * 1000.0 if (ts1_epoch and ts2_epoch) else None
                        queue_ms = (ts3_epoch - ts2_epoch) * 1000.0 if ts2_epoch else None
                        e2e_ms = (time.time() - ts1_epoch) * 1000.0 if ts1_epoch else None
                        PERFORMANCE_MONITOR.record_frame(
                            stream_key,
                            {
                                "capture": cap_ms,
                                "queue": queue_ms,
                                "redis_fetch": redis_fetch_ms,
                                "decode": decode_ms,
                                "detection": None,
                                "vllm_batch": None,
                                "processing": None,
                                "end_to_end": e2e_ms,
                            },
                            msg_id=msg_id,
                            frame_index=frame_idx,
                            persons=0,
                        )
                    continue

                frame_idx += 1
                if progress is not None:
                    progress.update(1)

                if frame_idx % frame_interval != 0:
                    continue

                # ======= 以下基本保持原来的检测 + Qwen 推理逻辑，只在最后增加 on_result 回调 =======
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                img_width, img_height = frame_pil.size

                t_detect_start = time.perf_counter()
                detections = self._detect_with_grounding(frame_pil)
                t_detect_end = time.perf_counter()
                detection_ms = (t_detect_end - t_detect_start) * 1000.0
                print(f"[stream {stream_key}] Frame {frame_idx}: Detected {len(detections)} persons")
                if not detections:
                    if MONITORING_ENABLED:
                        ts1_epoch = _parse_iso_to_epoch(meta.get("ts1")) if meta else None
                        ts2_epoch = _parse_iso_to_epoch(meta.get("ts2")) if meta else None
                        ts3_epoch = time.time()
                        cap_ms = (ts2_epoch - ts1_epoch) * 1000.0 if (ts1_epoch and ts2_epoch) else None
                        queue_ms = (ts3_epoch - ts2_epoch) * 1000.0 if ts2_epoch else None
                        processing_ms = (t_detect_end - decode_start) * 1000.0
                        e2e_ms = (time.time() - ts1_epoch) * 1000.0 if ts1_epoch else None
                        PERFORMANCE_MONITOR.record_frame(
                            stream_key,
                            {
                                "capture": cap_ms,
                                "queue": queue_ms,
                                "redis_fetch": redis_fetch_ms,
                                "decode": decode_ms,
                                "detection": detection_ms,
                                "vllm_batch": None,
                                "processing": processing_ms,
                                "end_to_end": e2e_ms,
                            },
                            msg_id=msg_id,
                            frame_index=frame_idx,
                            persons=0,
                        )
                    continue

                per_frame_results: List[Dict[str, Any]] = []
                person_jobs: List[Dict[str, Any]] = []
                qwen_http_times: List[float] = []
                qwen_total_times: List[float] = []
                qwen_parse_times: List[float] = []

                t_batch_start = time.perf_counter()

                for person_id, detection in enumerate(detections):
                    x1, y1, x2, y2 = _expand_box(
                        detection.box_xyxy, box_expand_ratio, img_width, img_height
                    )
                    crop = frame[y1:y2, x1:x2].copy()
                    crop, _, _ = _ensure_min_short_side(crop, crop_min_short_side)

                    crop_path = None
                    if save_root and save_crops:
                        crop_name = f"frame_{frame_idx:06d}_person_{person_id:02d}.jpg"
                        crop_path = str(save_root / "crops" / crop_name)
                        cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    future = self._vllm_executor.submit(
                        self._classify_action_with_qwen, crop
                    )

                    person_jobs.append(
                        {
                            "person_id": person_id,
                            "detection": detection,
                            "box": (x1, y1, x2, y2),
                            "crop_path": crop_path,
                            "future": future,
                        }
                    )

                # 收集每个人的 Qwen 结果
                detections_payload: List[Dict[str, Any]] = []

                for job in person_jobs:
                    person_id = job["person_id"]
                    detection = job["detection"]
                    x1, y1, x2, y2 = job["box"]
                    crop_path = job["crop_path"]
                    future = job["future"]

                    qwen_action: QwenAction = future.result()

                    if qwen_action.timings.get("http_ms") is not None:
                        qwen_http_times.append(float(qwen_action.timings["http_ms"]))
                    if qwen_action.timings.get("total_ms") is not None:
                        qwen_total_times.append(float(qwen_action.timings["total_ms"]))
                    if qwen_action.timings.get("parse_json_ms") is not None:
                        qwen_parse_times.append(float(qwen_action.timings["parse_json_ms"]))

                    per_frame_results.append(
                        {
                            "expanded_box": (x1, y1, x2, y2),
                            "preds": [
                                {
                                    "class_name": qwen_action.class_name,
                                    "confidence": qwen_action.confidence,
                                    "rationale": qwen_action.rationale,
                                }
                            ],
                        }
                    )

                    results.append(
                        PersonActionResult(
                            frame_index=frame_idx,
                            person_index=person_id,
                            box_xyxy=(x1, y1, x2, y2),
                            grounding_score=detection.score,
                            crop_path=crop_path,
                            qwen_action=qwen_action,
                        )
                    )

                    # === 组装 detections 里的一条记录（注意要归一化到 0-1） ===
                    bbox_xyxy_norm = [
                        x1 / img_width,
                        y1 / img_height,
                        x2 / img_width,
                        y2 / img_height,
                    ]
                    detections_payload.append(
                        {
                            "bbox_xyxy": bbox_xyxy_norm,
                            "conf": qwen_action.confidence if qwen_action.confidence is not None else float(detection.score),
                            "cls": qwen_action.class_name or "unknown",
                        }
                    )

                t_batch_end = time.perf_counter()
                batch_ms = (t_batch_end - t_batch_start) * 1000.0
                qwen_http_ms = max(qwen_http_times) if qwen_http_times else None
                qwen_total_ms = max(qwen_total_times) if qwen_total_times else None
                qwen_parse_ms = max(qwen_parse_times) if qwen_parse_times else None
                print(
                    f"[stream {stream_key}] vLLM batch for {len(person_jobs)} persons "
                    f"took {batch_ms:.2f} ms (workers={self._vllm_workers})"
                )

                # === 如果有 on_result 回调，就按接口文档格式组装一份消息 ===
                if on_result is not None and detections_payload:
                    try:
                        stream_url = meta.get("stream", stream_key)
                        payload = {
                            "stream": stream_url,
                            "msg_id": msg_id,
                            "meta": meta,
                            "detections": detections_payload,
                            "latency": {
                                # 这里只把我们掌握的一点信息塞进去，其它字段可按需扩展
                                "infer_ms": qwen_http_ms or 0.0,
                                "post_ms": qwen_parse_ms or 0.0,
                            },
                        }
                        on_result(payload)
                    except Exception as e:
                        print(f"[stream {stream_key}] on_result callback failed: {e}")

                has_detections = bool(per_frame_results)
                need_video_frame = save_root is not None and save_video
                need_image_frame = bool(save_root and save_anno and has_detections)

                if need_video_frame or need_image_frame:
                    annotated = frame.copy()
                    if has_detections:
                        for person_data in per_frame_results:
                            x1, y1, x2, y2 = person_data["expanded_box"]
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 200, 80), 2)
                            preds = person_data.get("preds", [])
                            for rank, pred in enumerate(preds[:top_k]):
                                label_text = pred.get("class_name") or "unknown"
                                conf = pred.get("confidence")
                                if conf is not None:
                                    label_text += f" {conf * 100:.1f}%"
                                y_anchor = max(15, y1 - 10 - rank * 18)
                                cv2.putText(
                                    annotated,
                                    label_text,
                                    (x1, y_anchor),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (50, 220, 50),
                                    2,
                                    lineType=cv2.LINE_AA,
                                )

                    if need_video_frame:
                        if writer is None:
                            if save_root is not None:
                                video_output_path = (save_root / annotated_video_name).with_suffix(".mp4")
                                video_output_path.parent.mkdir(parents=True, exist_ok=True)
                            frame_size = (annotated.shape[1], annotated.shape[0])
                            writer = _create_video_writer(video_output_path, fps, frame_size)
                        writer.write(annotated)

                    if need_image_frame and save_root is not None:
                        annotated_name = f"frame_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(save_root / "annotated_frames" / annotated_name), annotated)

                processing_end = time.perf_counter()
                processing_ms = (processing_end - decode_start) * 1000.0

                if MONITORING_ENABLED:
                    ts1_epoch = _parse_iso_to_epoch(meta.get("ts1")) if meta else None
                    ts2_epoch = _parse_iso_to_epoch(meta.get("ts2")) if meta else None
                    ts3_epoch = time.time()
                    cap_ms = (ts2_epoch - ts1_epoch) * 1000.0 if (ts1_epoch and ts2_epoch) else None
                    queue_ms = (ts3_epoch - ts2_epoch) * 1000.0 if ts2_epoch else None
                    e2e_ms = (time.time() - ts1_epoch) * 1000.0 if ts1_epoch else None
                    PERFORMANCE_MONITOR.record_frame(
                        stream_key,
                        {
                            "capture": cap_ms,
                            "queue": queue_ms,
                            "redis_fetch": redis_fetch_ms,
                            "decode": decode_ms,
                            "detection": detection_ms,
                            "vllm_batch": batch_ms,
                            "vllm_http": qwen_http_ms,
                            "vllm_parse": qwen_parse_ms,
                            "vllm_total": qwen_total_ms,
                            "processing": processing_ms,
                            "end_to_end": e2e_ms,
                        },
                        msg_id=msg_id,
                        frame_index=frame_idx,
                        persons=len(person_jobs),
                    )

        finally:
            if writer is not None:
                writer.release()
            if progress is not None:
                progress.close()

        return results


__all__ = [
    "QwenPersonActionPipeline",
    "PersonActionResult",
    "QwenAction",
    "GroundingResult",
]
