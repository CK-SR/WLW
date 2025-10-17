import time
import json
import cv2
import numpy as np
import asyncio
import concurrent.futures
from typing import Dict, Set, Optional
import redis
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import os
import re

import camera_check_fastapi.src.settings as settings # 导入自定义配置
from configs.settings import settings as config_settings

# === MinIO ===
import io
from minio import Minio
from minio.error import S3Error


EXECUTOR_CPU = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(8,os.cpu_count()*2)
)

EXECUTOR_IO = concurrent.futures.ThreadPoolExecutor(max_workers=64)

app = FastAPI(title="PushHub", version="1.0")

# 定义 Router
router = APIRouter(prefix="/pushhub", tags=["PushHub"])

r = redis.Redis(
    host=config_settings.redis.host,
    port=config_settings.redis.port,
    password=config_settings.redis.password,
    db=config_settings.redis.db,
    decode_responses=config_settings.redis.decode_responses,
)

# 默认参数
CURRENT_THRESHOLD = settings.DEFAULT_CURRENT_THRESHOLD
EDGE_PARAMS = settings.DEFAULT_EDGE_PARAMS.copy()
TAMPER_PARAMS = {
    "diff_threshold": 10.0,        # 帧均值差分阈值
    "roi_diff_threshold": 8.0,     # ROI 小范围均值阈值
    "high_ratio": 0.15,            # 高差异像素比例阈值
}

EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)

SUBSCRIBED_STREAMS: Set[str] = set()
SUBSCRIBE_EPOCH: int = 0
SUB_LOCK = asyncio.Lock()

WS_CLIENTS: "set[WebSocket]" = set()
WS_LOCK = asyncio.Lock()

broadcast_queue: asyncio.Queue = asyncio.Queue()
STREAM_WORKERS: Dict[str, asyncio.Task] = {}
PREV_FRAMES: Dict[str, np.ndarray] = {}

# 服务是否处于激活状态
SERVICE_ACTIVE: bool = False

# ★ 测试模式配置
TEST_MODE = os.getenv("ENABLE_TEST_VIDEO", "0") == "1"
TEST_VIDEO_FPS = float(os.getenv("TEST_VIDEO_FPS", "20"))
TEST_VIDEO_DIR = os.getenv("TEST_VIDEO_DIR", "./test_videos")
if TEST_MODE:
    os.makedirs(TEST_VIDEO_DIR, exist_ok=True)


# ===== MinIO 配置（环境变量可覆盖）=====
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "192.168.130.162:9100")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET     = os.getenv("MINIO_BUCKET", "camera-checkout")
MINIO_SECURE     = os.getenv("MINIO_SECURE", "0") == "1"
WS_SEND_MODE = os.getenv("WS_SEND_MODE", "all")  # all / abnormal
MINIO_MAX_FRAMES_PER_STREAM = int(os.getenv("MINIO_MAX_FRAMES_PER_STREAM", "1000"))

# 触发清理的上传间隔（每路累计 N 次上传再触发一次清理，降低 list_objects 频率）
MINIO_TRIM_INTERVAL = int(os.getenv("MINIO_TRIM_INTERVAL", "1000"))

# 是否在文件名加 intrussion 后缀；仅命名习惯，无功能差异
MINIO_USE_INTRUSSION_SUFFIX = os.getenv("MINIO_USE_INTRUSSION_SUFFIX", "1") == "1"

# 每路上传计数器，用于间隔触发清理
TRIM_COUNTER: Dict[str, int] = {}

# 保存策略：all / abnormal / sample / none
MINIO_SAVE_MODE  = os.getenv("MINIO_SAVE_MODE", "all")
MINIO_SAMPLE_FPS = float(os.getenv("MINIO_SAMPLE_FPS", "1"))  # sample 模式下的目标 FPS
MINIO_JPEG_Q     = int(os.getenv("MINIO_JPEG_QUALITY", "85"))

MINIO_CLIENT: Minio | None = None
LAST_MINIO_SAVE_TS: Dict[str, float] = {}  # 每路上次保存时间（sample 用）

# ★ 安全文件名生成器
def safe_filename(stream_name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', "_", stream_name)


def build_obj_key(stream_name: str, ts_ms: int) -> str:
    """
    构造对象键：{safe_id}/{13位零填充毫秒时间戳}[ _intrussion].jpg
    13位零填充能保证按 object_name 的字典序即时间序。
    """
    safe_id = safe_filename(stream_name)
    stamp = f"{ts_ms:013d}"  # 始终 13 位，例如 001723... 保证字典序=时间序
    tail = "_intrussion" if MINIO_USE_INTRUSSION_SUFFIX else ""
    return f"{safe_id}/{stamp}{tail}.jpg"

from datetime import timezone, datetime

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso_to_epoch(iso_or_none) -> float | None:
    """把 '2025-10-10T02:13:19.584Z' 转为 epoch 秒(float)。异常返回 None。"""
    if not iso_or_none:
        return None
    try:
        s = str(iso_or_none).replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


# ★ numpy 转 Python 原生类型
def _to_builtin(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return o


def _minio_init():
    """初始化 MinIO 客户端 & 确保 bucket 存在（启动时调用一次）"""
    global MINIO_CLIENT
    try:
        MINIO_CLIENT = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        if not MINIO_CLIENT.bucket_exists(MINIO_BUCKET):
            MINIO_CLIENT.make_bucket(MINIO_BUCKET)
        print(f"[minio] ready -> {MINIO_ENDPOINT} bucket={MINIO_BUCKET}")
    except Exception as e:
        MINIO_CLIENT = None
        print(f"[minio][error] init failed: {e}")

async def minio_put_bytes(obj_key: str, data: bytes, content_type: str = "image/jpeg") -> str | None:
    """把 bytes 异步上传到 MinIO，返回 etag；失败返回 None。放到 IO 线程池执行，避免阻塞事件循环。"""
    if not MINIO_CLIENT:
        return None
    loop = asyncio.get_running_loop()
    bio = io.BytesIO(data)
    size = len(data)

    def _put():
        try:
            result = MINIO_CLIENT.put_object(
                MINIO_BUCKET, obj_key, bio, size, content_type=content_type
            )
            return getattr(result, "etag", None)
        except S3Error as e:
            print(f"[minio][S3Error] put {obj_key}: {e}")
            return None
        except Exception as e:
            print(f"[minio][error] put {obj_key}: {e}")
            return None

    return await loop.run_in_executor(EXECUTOR_IO, _put)


async def minio_trim_prefix(prefix: str, keep_last_n: int) -> int:
    """保留前缀下最近 keep_last_n 个对象，多余的按时间顺序删除；返回删除数量"""
    if not MINIO_CLIENT or keep_last_n <= 0:
        return 0

    loop = asyncio.get_running_loop()

    def _trim():
        try:
            objs = list(MINIO_CLIENT.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True))
            total = len(objs)
            if total <= keep_last_n:
                # 快照日志
                print(f"[minio][trim][snapshot] prefix={prefix} total={total} (<= keep)")
                return 0
            # 如果对象名里使用了 13 位零填充的毫秒时间戳（build_obj_key)，
            # 可直接按 object_name 排序代表时间先后；否则按 last_modified。
            try:
                objs_sorted = sorted(objs, key=lambda o: o.object_name)
            except Exception:
                objs_sorted = sorted(objs, key=lambda o: o.last_modified)

            to_del = objs_sorted[: total - keep_last_n]  # 删除更旧的
            cnt = 0
            for o in to_del:
                try:
                    MINIO_CLIENT.remove_object(MINIO_BUCKET, o.object_name)
                    cnt += 1
                except Exception as e:
                    print(f"[minio][trim][warn] remove {o.object_name}: {e}")

            # 再做一次快照
            rest = total - cnt
            newest = objs_sorted[-1].object_name if rest > 0 else "-"
            oldest_after = objs_sorted[total - cnt].object_name if rest > 0 else "-"
            print(f"[minio][trim][snapshot] prefix={prefix} deleted={cnt} rest={rest} oldest_after={oldest_after} newest={newest}")
            return cnt
        except Exception as e:
            print(f"[minio][trim][error] {e}")
            return 0

    return await loop.run_in_executor(EXECUTOR_IO, _trim)

# ★★★ 替换原函数签名和所有 return ★★★
def _decode_and_analyze_frame(
    jpeg_bytes: bytes,
    prev_frame: Optional[np.ndarray] = None,
    current_threshold: int = CURRENT_THRESHOLD,
    edge_params: Dict = EDGE_PARAMS,
    tamper_params: Dict = TAMPER_PARAMS,
) -> tuple[dict, Optional[np.ndarray]]:
    try:
        t0 = time.monotonic()
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        t1 = time.monotonic()
        decode_ms = round((t1 - t0) * 1000, 3)

        if frame is None:
            return ({"state": "Error", "error": "Decoding failed", "decode_ms": decode_ms}, prev_frame)

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 黑屏
        if float(grey.mean()) < float(current_threshold):
            return ({"state": "Black Screen", "decode_ms": decode_ms}, grey)

        # 2. 遮挡
        edges = cv2.Canny(grey, int(edge_params["low"]), int(edge_params["high"]))
        edge_ratio = float(cv2.countNonZero(edges)) / float(grey.size)
        if edge_ratio < float(edge_params["min_ratio"]):
            return ({"state": "Occluded", "edge_ratio": round(edge_ratio * 100.0, 2), "decode_ms": decode_ms}, grey)

        # 3. 恶意破坏
        if prev_frame is not None:
            diff = cv2.absdiff(grey, prev_frame)
            diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
            h, w = diff_blur.shape
            grid_h, grid_w = h // 3, w // 3
            roi_changes = 0
            for i in range(3):
                for j in range(3):
                    roi = diff_blur[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                    if roi.mean() > float(tamper_params["roi_diff_threshold"]):
                        roi_changes += 1
            roi_ratio = roi_changes / 9.0
            diff_mean = float(diff_blur.mean())
            high_diff_ratio = float(np.sum(diff_blur > 25)) / float(diff_blur.size)

            if (diff_mean > float(tamper_params["diff_threshold"]) and
                (high_diff_ratio > float(tamper_params["high_ratio"]) or roi_ratio > 0.5)):
                return ({
                    "state": "Tampered (Violent Motion)",
                    "diff_mean": round(diff_mean, 2),
                    "high_diff_ratio": round(high_diff_ratio, 2),
                    "roi_ratio": round(roi_ratio, 2),
                    "decode_ms": decode_ms
                }, grey)

        # 4. 正常
        return ({"state": "Normal", "edge_ratio": round(edge_ratio * 100.0, 2), "decode_ms": decode_ms}, grey)

    except Exception as e:
        print(f"[analyze][error] {e}")
        return ({"state": "Error", "error": str(e), "decode_ms": 0.0}, prev_frame)


# ★★★ 替换 analyze_frame_async 返回类型 ★★★
async def analyze_frame_async(jpeg_bytes: bytes, prev_frame: Optional[np.ndarray]) -> tuple[dict, Optional[np.ndarray]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        EXECUTOR_CPU,
        _decode_and_analyze_frame,
        jpeg_bytes,
        prev_frame,
        CURRENT_THRESHOLD,
        EDGE_PARAMS,
        TAMPER_PARAMS,
    )


def _read_one_from_redis(stream_name: str, block_ms: int = 1000):
    return r.xread({f"frames:{stream_name}": "$"}, count=1, block=block_ms) or []


async def get_one_frame(stream_name: str, timeout_sec: float = 1.0):
    loop = asyncio.get_running_loop()
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        streams = await loop.run_in_executor(
            EXECUTOR_IO, _read_one_from_redis, stream_name, int(timeout_sec * 1000)
        )
        if streams:
            print(f"[redis] got frame for {stream_name}")
            for _stream, messages in streams:
                for _id, fields in messages:
                    meta = json.loads(fields[b"meta"].decode("utf-8"))
                    jpeg = fields[b"jpeg"]
                    return meta, jpeg
        else:
            print(f"[redis] no frame for {stream_name}, retry...")
        await asyncio.sleep(0.01)
    return None, None


async def stream_worker(stream_name: str):
    print(f"[worker start] {stream_name}")

    video_writer = None
    frame_count = 0

    try:
        while True:
            # ========== 获取帧 ==========
            meta, jpeg = await get_one_frame(stream_name, timeout_sec=1.0)
            if jpeg is None:
                continue

            ts3_epoch = time.time()
            ts3_iso = now_iso_utc()

            # ========== 调用分析（仅在 CPU 线程里解码一次）==========
            prev_frame = PREV_FRAMES.get(stream_name)
            t_proc_start = time.monotonic()
            analysis, curr_grey = await analyze_frame_async(jpeg, prev_frame)
            t_proc_end = time.monotonic()
            if curr_grey is not None:
                PREV_FRAMES[stream_name] = curr_grey

            # ========== 计算时延 ==========
            ts1_epoch = parse_iso_to_epoch(meta.get("ts1"))
            ts2_epoch = parse_iso_to_epoch(meta.get("ts2"))
            cap_ms   = (ts2_epoch - ts1_epoch) * 1000.0 if (ts1_epoch and ts2_epoch) else None
            queue_ms = (ts3_epoch - ts2_epoch) * 1000.0 if ts2_epoch else None
            proc_ms  = (t_proc_end - t_proc_start) * 1000.0
            decode_ms   = float(analysis.get("decode_ms", 0.0))
            analysis_ms = max(proc_ms - decode_ms, 0.0)
            ts4_iso  = now_iso_utc()
            e2e_ms   = (time.time() - ts1_epoch) * 1000.0 if ts1_epoch else None

            # ========== 异常判定 ==========
            is_abnormal = analysis.get("state") not in ("Normal", None)

            # ========== MinIO: 仍然写“原始 JPEG”，不做叠加/再编码 ==========
            raw_jpeg = jpeg
            ts_ms = int(time.time() * 1000)
            obj_key = build_obj_key(stream_name, ts_ms)

            # 保存策略：若你只想异常时写 -> 环境变量设 MINIO_SAVE_MODE=abnormal
            save_this = (MINIO_SAVE_MODE == "all") or (MINIO_SAVE_MODE == "abnormal" and is_abnormal)
            _minio_pending_ref = None
            if save_this and MINIO_CLIENT:
                etag = await minio_put_bytes(obj_key, raw_jpeg, content_type="image/jpeg")
                if etag:
                    _minio_pending_ref = {
                        "type": "frame",
                        "key": obj_key,
                        "minio_object_key": obj_key,
                        "etag": etag,
                        "reason": "abnormal" if is_abnormal else "all"
                    }
                    # 分批触发清理
                    safe_id = safe_filename(stream_name)
                    cnt = TRIM_COUNTER.get(safe_id, 0) + 1
                    TRIM_COUNTER[safe_id] = cnt
                    if cnt % MINIO_TRIM_INTERVAL == 0:
                        deleted = await minio_trim_prefix(prefix=f"{safe_id}/", keep_last_n=MINIO_MAX_FRAMES_PER_STREAM)
                        print(f"[minio][trim] trigger prefix={safe_id}/ deleted={deleted} keep={MINIO_MAX_FRAMES_PER_STREAM}")

            # ========== 组织 WS payload（仍包含分析结果）==========
            payload = {
                "stream": str(stream_name),
                "ts_wall": meta.get("ts_wall_iso"),
                "shape": meta.get("shape"),
                "result": analysis,  # 这里就是你的分析结果
                "timing": {
                    "ts1": meta.get("ts1"),
                    "ts2": meta.get("ts2"),
                    "ts3": ts3_iso,
                    "ts4": ts4_iso,
                    "lat_ms": {
                        "cap": None if cap_ms is None else round(cap_ms, 3),
                        "queue": None if queue_ms is None else round(queue_ms, 3),
                        "decode": round(decode_ms, 3),
                        "analysis": round(analysis_ms, 3),
                        "proc": round(proc_ms, 3),
                        "end2end": None if e2e_ms is None else round(e2e_ms, 3),
                    }
                }
            }
            if _minio_pending_ref:
                payload.setdefault("obj_refs", []).append(_minio_pending_ref)

            # WS 发送开关：all / abnormal
            if WS_SEND_MODE == "abnormal" and not is_abnormal:
                pass
            else:
                await broadcast_queue.put(payload)

    except asyncio.CancelledError:
        print(f"[worker stop] {stream_name}")
        raise
    except Exception as e:
        print(f"[worker error] {stream_name}: {e}")
        await asyncio.sleep(0.5)
    finally:
        if video_writer is not None and video_writer.isOpened():
            video_writer.release()
            print(f"[worker] saved video for {stream_name}, total frames={frame_count}")


async def broadcaster():
    print("[broadcaster] running")
    ticks = 0
    while True:
        payload = await broadcast_queue.get()
        stream = payload["stream"]

        async with SUB_LOCK:
            if stream not in SUBSCRIBED_STREAMS:
                continue
            epoch = SUBSCRIBE_EPOCH

        message = {"epoch": int(epoch), "data": payload}

        try:
            text = json.dumps(message, ensure_ascii=False, default=_to_builtin)
        except Exception as e:
            try:
                text = json.dumps(_to_builtin(message), ensure_ascii=False, default=_to_builtin)
            except Exception as e2:
                print(f"[broadcast][serialize-error] {e} / {e2}, drop message")
                continue

        async with WS_LOCK:
            targets = list(WS_CLIENTS)

        if not targets:
            continue

        for ws in list(targets):
            try:
                await ws.send_text(text)
            except Exception as e:
                print(f"[broadcast][error] ws send failed: {e}")
                async with WS_LOCK:
                    WS_CLIENTS.discard(ws)

        ticks += 1
        if ticks % 100 == 0:
            print(f"[broadcast] sent {ticks} msgs, active_ws={len(WS_CLIENTS)}")


@router.post("/start")
async def start_service():
    global SERVICE_ACTIVE
    if SERVICE_ACTIVE:
        print("[service] start called, but already active")
        return {"ok": True, "msg": "Service already started"}
    SERVICE_ACTIVE = True
    print("[service] started")
    return {"ok": True, "msg": "Service started"}


@router.post("/stop")
async def stop_service():
    global SERVICE_ACTIVE, STREAM_WORKERS, SUBSCRIBED_STREAMS, PREV_FRAMES
    if not SERVICE_ACTIVE:
        print("[service] stop called, but already inactive")
        return {"ok": True, "msg": "Service already stopped"}

    SERVICE_ACTIVE = False
    print("[service] stopping workers...")
    for name, task in list(STREAM_WORKERS.items()):
        print(f"[service] cancelling worker {name}")
        task.cancel()
    STREAM_WORKERS.clear()
    SUBSCRIBED_STREAMS.clear()
    PREV_FRAMES.clear()
    print("[service] stopped, resources released")
    return {"ok": True, "msg": "Service stopped and resources released"}


@router.post("/subscribe")
async def subscribe(body: Dict = Body(...)):
    global CURRENT_THRESHOLD, EDGE_PARAMS, TAMPER_PARAMS

    if not SERVICE_ACTIVE:
        print("[subscribe] rejected, service not started")
        return JSONResponse({"ok": False, "error": "Service not started"}, status_code=400)

    streams = body.get("streams", [])
    if not isinstance(streams, list):
        return JSONResponse({"ok": False, "error": "streams must be a list"}, status_code=400)

    CURRENT_THRESHOLD = int(body.get("current_threshold", settings.DEFAULT_CURRENT_THRESHOLD))
    EDGE_PARAMS.update({k: v for k, v in body.get("edge_params", {}).items() if k in EDGE_PARAMS})
    TAMPER_PARAMS.update({k: v for k, v in body.get("tamper_params", {}).items() if k in TAMPER_PARAMS})

    async with SUB_LOCK:
        global SUBSCRIBE_EPOCH, SUBSCRIBED_STREAMS
        SUBSCRIBED_STREAMS = set(map(str, streams))
        SUBSCRIBE_EPOCH += 1
        curr_epoch = SUBSCRIBE_EPOCH

    for name in SUBSCRIBED_STREAMS:
        if name not in STREAM_WORKERS:
            print(f"[subscribe] creating worker for {name}")
            STREAM_WORKERS[name] = asyncio.create_task(stream_worker(name))

    print(f"[subscribe] epoch={curr_epoch}, subscribed={list(SUBSCRIBED_STREAMS)}")
    print(f"[params] current_threshold={CURRENT_THRESHOLD}, edge_params={EDGE_PARAMS}, tamper_params={TAMPER_PARAMS}")
    return {"ok": True, "epoch": curr_epoch, "subscribed": list(SUBSCRIBED_STREAMS)}


@router.websocket("/ws/results")
async def ws_results(websocket: WebSocket):
    await websocket.accept()
    async with WS_LOCK:
        WS_CLIENTS.add(websocket)
    print("[ws] client connected, total=", len(WS_CLIENTS))
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[ws] client disconnected")
    finally:
        async with WS_LOCK:
            WS_CLIENTS.discard(websocket)
            print("[ws] client removed, total=", len(WS_CLIENTS))


@router.on_event("startup")
async def on_start():
    asyncio.create_task(broadcaster())
    _minio_init()  


# 挂载 router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5020, log_level="info")
