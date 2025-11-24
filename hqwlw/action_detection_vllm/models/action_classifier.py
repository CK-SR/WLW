from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class QwenAction:
    class_id: Optional[int]
    class_name: Optional[str]
    confidence: Optional[float]
    rationale: Optional[str]
    raw_response: str
    timings: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRequest:
    image: np.ndarray
    meta: Dict[str, Any]
    future: asyncio.Future[QwenAction]


def _img_ndarray_to_data_url(image: np.ndarray, quality: int = 95) -> str:
    from PIL import Image
    import base64
    import io

    rgb = image[:, :, ::-1]
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


class ActionClassifier:
    def __init__(
        self,
        batch_size: int = 8,
        queue_size: int = 512,
    ) -> None:
        self.batch_size = batch_size
        self.queue: asyncio.Queue[ActionRequest] = asyncio.Queue(maxsize=queue_size)
        self._workers: List[asyncio.Task[None]] = []
        self._stopped = asyncio.Event()

    async def start(self, num_workers: Optional[int] = None) -> None:
        if self._workers:
            return
        worker_count = num_workers or settings.vllm.max_workers
        for _ in range(worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop()))

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._workers:
            task.cancel()
        self._workers.clear()

    async def submit(self, image: np.ndarray, meta: Dict[str, Any]) -> QwenAction:
        future: asyncio.Future[QwenAction] = asyncio.get_running_loop().create_future()
        req = ActionRequest(image=image, meta=meta, future=future)
        await self.queue.put(req)
        return await future

    async def _worker_loop(self) -> None:
        session_timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            while not self._stopped.is_set():
                batch: List[ActionRequest] = []
                try:
                    req = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                    batch.append(req)
                except asyncio.TimeoutError:
                    continue

                while len(batch) < self.batch_size:
                    try:
                        req = self.queue.get_nowait()
                        batch.append(req)
                    except asyncio.QueueEmpty:
                        break

                await self._process_batch(batch, session)

    async def _process_batch(self, batch: List[ActionRequest], session: aiohttp.ClientSession) -> None:
        t0 = time.perf_counter()
        images = [_img_ndarray_to_data_url(req.image) for req in batch]

        user_text = (
            "You will receive multiple person crops. "
            "Return a JSON list where each item matches the order of incoming images. "
            '{"class_id": int, "class_name": str, "confidence": float, "rationale": str}."'
        )

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": img}})

        messages = []
        if settings.prompt.system_path:
            try:
                system_text = Path(settings.prompt.system_path).read_text(encoding="utf-8")
            except Exception:
                system_text = ""
            if system_text:
                messages.append({"role": "system", "content": system_text})

        messages.append({"role": "user", "content": content})

        payload = {
            "model": settings.vllm.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 256,
        }

        try:
            async with session.post(
                settings.vllm.base_url.rstrip("/") + "/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer EMPTY"},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            logger.exception("vLLM request failed: %s", exc)
            for req in batch:
                if not req.future.done():
                    req.future.set_result(
                        QwenAction(
                            class_id=None,
                            class_name=None,
                            confidence=None,
                            rationale=None,
                            raw_response=str(exc),
                            timings={},
                            meta=req.meta,
                        )
                    )
            return

        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            joined = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
            content_text = joined
        else:
            content_text = str(content)

        try:
            results = json.loads(content_text)
        except Exception:
            results = []

        if not isinstance(results, list):
            results = [results]

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        for idx, req in enumerate(batch):
            item = results[idx] if idx < len(results) else {}
            action = QwenAction(
                class_id=item.get("class_id") if isinstance(item, dict) else None,
                class_name=item.get("class_name") if isinstance(item, dict) else None,
                confidence=item.get("confidence") if isinstance(item, dict) else None,
                rationale=item.get("rationale") if isinstance(item, dict) else None,
                raw_response=content_text,
                timings={"http_ms": total_ms, "total_ms": total_ms, "batch_size": len(batch)},
                meta=req.meta,
            )
            if not req.future.done():
                req.future.set_result(action)
