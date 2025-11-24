from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel

from config.settings import settings
from models.action_classifier import ActionClassifier
from models.detection import PersonDetector
from pipeline.core import PosePipeline
from service.ws import broadcast, router as ws_router

app = FastAPI(title="异常行为检测", version="v2")
router = APIRouter(prefix="/pose_pipeline", tags=["pose-pipeline"])


class StartRequest(BaseModel):
    discover_all: bool = False
    stream_prefix: str = settings.redis.stream_prefix
    cameras: List[str] = []


class SubStatus(BaseModel):
    running: bool = False


class PipelineStatus(BaseModel):
    running: bool = False
    framesource: SubStatus = SubStatus()
    inference: SubStatus = SubStatus()
    resultserver: SubStatus = SubStatus()


class PosePipelineManager:
    def __init__(self) -> None:
        self.running = False
        self.stream_prefix = settings.redis.stream_prefix
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._tasks: Dict[str, asyncio.Future[Any]] = {}
        detector = PersonDetector(
            model_path=settings.yolo.model_path,
            device=settings.yolo.device,
            conf=settings.yolo.conf,
        )
        classifier = ActionClassifier(batch_size=8, queue_size=1024)
        self.pipeline = PosePipeline(detector=detector, classifier=classifier)

    def _ensure_loop(self) -> None:
        if self._loop and self._loop_thread:
            return
        self._loop = asyncio.new_event_loop()

        def runner() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=runner, daemon=True)
        self._loop_thread.start()
        asyncio.run_coroutine_threadsafe(
            self.pipeline.classifier.start(settings.vllm.max_workers), self._loop
        )

    def _schedule_stream(self, stream_key: str) -> None:
        assert self._loop
        safe_name = stream_key.replace(":", "_").replace("/", "_")
        out_dir = Path(f"./outputs/qwen_vllm_stream/{safe_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        def emit(result: Dict[str, Any]) -> None:
            if self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(broadcast(result), self._loop)

        coro = self.pipeline.process_stream_async(
            stream_key=stream_key,
            output_dir=str(out_dir),
            on_result=emit,
            manage_classifier=False,
        )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        self._tasks[stream_key] = future

    def start(self, req: StartRequest) -> None:
        from redis import Redis

        with self._lock:
            if self.running:
                return
            self.running = True
            self.stream_prefix = req.stream_prefix
            self._ensure_loop()

            if req.discover_all:
                r = Redis(
                    host=settings.redis.host,
                    port=settings.redis.port,
                    password=None,
                    db=0,
                    decode_responses=False,
                )
                cameras: List[str] = []
                for key in r.scan_iter(f"{self.stream_prefix}*"):
                    k = key.decode() if isinstance(key, bytes) else key
                    cameras.append(k)
            else:
                cameras = req.cameras

            for stream_key in cameras:
                if stream_key in self._tasks:
                    continue
                self.pipeline.request_stop(stream_key)
                self.pipeline._stop_flags[stream_key] = False
                self._schedule_stream(stream_key)

    def stop(self) -> None:
        with self._lock:
            self.running = False
            self.pipeline.request_stop()
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    self.pipeline.classifier.stop(), self._loop
                )
                for future in self._tasks.values():
                    future.cancel()
                self._tasks.clear()
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop = None
                self._loop_thread = None

    def get_status(self) -> PipelineStatus:
        framesource_running = self.running and bool(self._tasks)
        inference_running = framesource_running
        resultserver_running = False
        return PipelineStatus(
            running=framesource_running or inference_running or resultserver_running,
            framesource=SubStatus(running=framesource_running),
            inference=SubStatus(running=inference_running),
            resultserver=SubStatus(running=resultserver_running),
        )


manager = PosePipelineManager()


@router.get("/pipeline/status", response_model=PipelineStatus)
async def pipeline_status() -> PipelineStatus:
    return manager.get_status()


@router.post("/pipeline/start", response_model=PipelineStatus)
async def pipeline_start(req: StartRequest) -> PipelineStatus:
    manager.start(req)
    return manager.get_status()


@router.post("/pipeline/stop", response_model=PipelineStatus)
async def pipeline_stop() -> PipelineStatus:
    manager.stop()
    return manager.get_status()


app.include_router(router)
app.include_router(ws_router)
