# pose_pipeline_api.py
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from action_detection_vllm.run_qwen_action_pipeline_vllm import create_pipeline
from action_detection_vllm.pipeline_qwen_lzy import set_upload_all_frames, set_upload_annotated

app = FastAPI(title="异常行为检测", version="v1")
router = APIRouter(prefix="/pose_pipeline", tags=["pose-pipeline"])

# 全局 pipeline 实例
PIPELINE = create_pipeline()


# ====== WebSocket 客户端管理 ======
clients: Set[WebSocket] = set()
clients_lock = asyncio.Lock()


async def broadcast(message: Dict[str, Any]) -> None:
    """把检测结果推送给所有已连接的 WebSocket 客户端"""
    dead: List[WebSocket] = []
    async with clients_lock:
        for ws in list(clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.remove(ws)


# ====== Pipeline Manager：管理多路流、线程与状态 ======
class StartRequest(BaseModel):
    discover_all: bool = False
    stream_prefix: str = "frames:"
    cameras: List[str] = []   # 例如 ["frames:rtsp://192.168.1.10:8554/cam01", ...]


class UploadModeRequest(BaseModel):
    upload_all_frames: bool


class AnnotatedModeRequest(BaseModel):
    upload_annotated: bool


class SubStatus(BaseModel):
    running: bool = False


class PipelineStatus(BaseModel):
    running: bool = False
    framesource: SubStatus = SubStatus()
    inference: SubStatus = SubStatus()
    resultserver: SubStatus = SubStatus()


class PosePipelineManager:
    def __init__(self):
        self.running: bool = False
        self.stream_prefix: str = "frames:"
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _run_one_stream(self, stream_key: str) -> None:
        """
        在线程中跑一条 Redis Stream。
        stream_key 如：frames:rtsp://192.168.1.10:8554/cam01
        """
        safe_name = stream_key.replace(":", "_").replace("/", "_")
        out_dir = Path(f"./outputs/qwen_vllm_stream/{safe_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        def emit(result: Dict[str, Any]) -> None:
            """被 pipeline 调用的回调函数，用来把结果抛到 WebSocket"""
            if self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(broadcast(result), self._loop)

        try:
            PIPELINE.process_stream(
                stream_key=stream_key,
                output_dir=str(out_dir),
                frame_interval=1,
                box_expand_ratio=0.4,
                crop_min_short_side=256,
                save_crops=False,
                save_anno=False,
                save_video=False,
                top_k=1,
                annotated_video_name="annotated_qwen",
                max_frames=None,           # 一直跑，直到 /stop
                timeout_sec=1.0,
                on_result=emit,
            )
        finally:
            # 跑完之后自动从 threads 中移除
            with self._lock:
                self._threads.pop(stream_key, None)
                if not self._threads:
                    self.running = False

    def start(self, req: StartRequest) -> None:
        """
        根据 discover_all / stream_prefix / cameras 创建线程并启动。
        """
        from action_detection_vllm.pipeline_qwen_lzy import r  # 用你原来的 redis 连接

        with self._lock:
            if self.running:
                # 已经在跑，就不重复启动；你也可以选择先 stop 再 start
                return
            self.running = True
            self.stream_prefix = req.stream_prefix

            # 决定要跑哪些 stream_key
            if req.discover_all:
                # 自动发现：扫描 redis 中的 frames:* key
                pattern = f"{self.stream_prefix}*"
                cameras: List[str] = []
                for key in r.scan_iter(pattern):
                    k = key.decode() if isinstance(key, bytes) else key
                    cameras.append(k)
            else:
                cameras = req.cameras

            # 每一路启动一个线程
            for stream_key in cameras:
                if stream_key in self._threads:
                    continue
                PIPELINE.request_stop(stream_key)  # 重置 stop flag
                PIPELINE._stop_flags[stream_key] = False
                t = threading.Thread(
                    target=self._run_one_stream,
                    args=(stream_key,),
                    daemon=True,
                )
                self._threads[stream_key] = t
                t.start()

    def stop(self) -> None:
        """
        通知所有流水线停止。真正退出要等各线程在下一轮循环检查到 stop_flag。
        """
        with self._lock:
            self.running = False
            PIPELINE.request_stop()  # 通知所有 stream 停止

    def get_status(self) -> PipelineStatus:
        # framesource / inference 按是否有线程在跑来判断
        framesource_running = self.running and bool(self._threads)
        inference_running = framesource_running
        # resultserver 根据是否有 WebSocket 客户端判断
        resultserver_running = len(clients) > 0
        return PipelineStatus(
            running=framesource_running or inference_running or resultserver_running,
            framesource=SubStatus(running=framesource_running),
            inference=SubStatus(running=inference_running),
            resultserver=SubStatus(running=resultserver_running),
        )


manager = PosePipelineManager()


# ====== FastAPI 生命周期：记录主事件循环 ======
@app.on_event("startup")
async def on_startup() -> None:
    loop = asyncio.get_running_loop()
    manager.set_loop(loop)


# ====== 1. 状态检查 GET /pose_pipeline/pipeline/status ======
@router.get("/pipeline/status", response_model=PipelineStatus)
async def pipeline_status() -> PipelineStatus:
    return manager.get_status()


# ====== 2. 开启异常行为检测任务 POST /pose_pipeline/pipeline/start ======
@router.post("/pipeline/start", response_model=PipelineStatus)
async def pipeline_start(req: StartRequest):
    """
    请求体示例：
    {
      "discover_all": false,
      "stream_prefix": "frames:",
      "cameras": [
        "frames:rtsp://192.168.1.10:8554/cam01",
        "frames:rtsp://192.168.1.11:8554/cam02"
      ]
    }
    """
    manager.start(req)
    return manager.get_status()


# ====== 3. 停止任务 POST /pose_pipeline/pipeline/stop ======
@router.post("/pipeline/stop", response_model=PipelineStatus)
async def pipeline_stop():
    manager.stop()
    return manager.get_status()


# ====== 4. 上传策略切换 ======
@router.post("/pipeline/upload_mode")
async def update_upload_mode(req: UploadModeRequest):
    """切换是否上传全量帧。"""
    return set_upload_all_frames(req.upload_all_frames)


@router.post("/pipeline/annotation_mode")
async def update_annotation_mode(req: AnnotatedModeRequest):
    """切换上传的帧是否带标注。"""
    return set_upload_annotated(req.upload_annotated)


# ====== 5. WebSocket /pose_pipeline/ws ======
@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with clients_lock:
        clients.add(websocket)
    try:
        while True:
            # 这里我们不要求客户端必须发消息，简单地保持连接即可
            await websocket.receive_text()
    except WebSocketDisconnect:
        async with clients_lock:
            if websocket in clients:
                clients.remove(websocket)


app.include_router(router)