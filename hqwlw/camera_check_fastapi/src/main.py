import time
import json
import cv2
import numpy as np
import asyncio
import concurrent.futures
from enum import Enum
from typing import Any, Callable, Dict, List, Set, Optional
import functools
import redis
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import os
import re
from uuid import uuid4

import camera_check_fastapi.src.settings as settings # 导入自定义配置
from configs.settings import settings as config_settings
from camera_check_fastapi.src.i18n import (
    DEFAULT_TRANSLATIONS,
    I18NConfig,
    LanguageManager,
)
from camera_check_fastapi.src.performance import PERFORMANCE_MONITOR

# === MinIO ===
import io
from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error


EXECUTOR_CPU = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(8,os.cpu_count()*2)
)

EXECUTOR_IO = concurrent.futures.ThreadPoolExecutor(max_workers=64)
MINIO_UPLOAD_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=32)
MINIO_TRIM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=16)

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

i18n_config = I18NConfig(
    default_language=config_settings.i18n.default_language,
    fallback_language=config_settings.i18n.fallback_language,
    supported_languages=tuple(config_settings.i18n.supported_languages or ["en"]),
)
LANGUAGE_MANAGER = LanguageManager(DEFAULT_TRANSLATIONS, config=i18n_config)

MONITORING_ENABLED = bool(config_settings.monitoring.enabled)
if MONITORING_ENABLED:
    PERFORMANCE_MONITOR.configure(
        history_size=int(config_settings.monitoring.history_size),
        frame_history=int(config_settings.monitoring.frame_history),
    )
else:
    PERFORMANCE_MONITOR.reset()

# 默认参数
CURRENT_THRESHOLD = settings.DEFAULT_CURRENT_THRESHOLD
EDGE_PARAMS = settings.DEFAULT_EDGE_PARAMS.copy()
TAMPER_PARAMS = {
    "diff_threshold": 10.0,        # 帧均值差分阈值
    "roi_diff_threshold": 8.0,     # ROI 小范围均值阈值
    "high_ratio": 0.15,            # 高差异像素比例阈值
}


class StreamState(str, Enum):
    NORMAL = "Normal"
    BLACK_SCREEN = "Black Screen"
    OCCLUDED = "Occluded"
    TAMPERED = "Tampered (Violent Motion)"
    ERROR = "Error"


STATE_KEY_MAP: Dict[StreamState, str] = {
    StreamState.NORMAL: "state.normal",
    StreamState.BLACK_SCREEN: "state.black_screen",
    StreamState.OCCLUDED: "state.occluded",
    StreamState.TAMPERED: "state.tampered",
    StreamState.ERROR: "state.error",
}


def localized_message_payload(key: str, field_name: str = "msg", **fmt: Any) -> Dict[str, Any]:
    message = LANGUAGE_MANAGER.message(key, **fmt)
    return {field_name: message["msg"], f"{field_name}_i18n": message["i18n"]}


def build_state_payload(
    state: StreamState,
    decode_ms: Optional[float],
    **extra: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"state": state.value}
    if decode_ms is not None:
        payload["decode_ms"] = round(float(decode_ms), 3)
    payload.update(extra)
    payload["state_i18n"] = LANGUAGE_MANAGER.bundle(STATE_KEY_MAP[state])
    return payload


def make_error_response(key: str, status_code: int = 400, **fmt: Any) -> JSONResponse:
    body = {"ok": False, **localized_message_payload(key, field_name="error", **fmt)}
    return JSONResponse(body, status_code=status_code)

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


# ===== MinIO 配置（配置文件 + 环境变量可覆盖）=====
MINIO_ENDPOINT = config_settings.minio.endpoint
MINIO_ACCESS_KEY = config_settings.minio.access_key
MINIO_SECRET_KEY = config_settings.minio.secret_key
MINIO_BUCKET = config_settings.minio.bucket
MINIO_SECURE = bool(config_settings.minio.secure)
MINIO_RING_ENABLED = bool(getattr(config_settings.minio, "ring_enabled", False))
MINIO_MAX_FRAMES_PER_STREAM = int(config_settings.minio.max_frames_per_stream)

# 触发清理的上传间隔（每路累计 N 次上传再触发一次清理，降低 list_objects 频率）
MINIO_TRIM_INTERVAL = int(config_settings.minio.trim_interval)

MINIO_TRIM_BATCH = int(os.getenv("MINIO_TRIM_BATCH", "1000"))
MINIO_TRIM_MAX_DELETE = int(os.getenv("MINIO_TRIM_MAX_DELETE", "2000"))

# 是否在文件名加 intrussion 后缀；仅命名习惯，无功能差异
MINIO_USE_INTRUSSION_SUFFIX = bool(config_settings.minio.use_intrussion_suffix)

# 保存策略：all / abnormal / sample / none
MINIO_SAVE_MODE = config_settings.minio.save_mode
MINIO_SAMPLE_FPS = float(config_settings.minio.sample_fps)  # sample 模式下的目标 FPS
MINIO_JPEG_Q = int(config_settings.minio.jpeg_quality)

WS_SEND_MODE = config_settings.stream.ws_send_mode  # all / abnormal

LAST_MINIO_SAVE_TS: Dict[str, float] = {}  # 每路上次保存时间（sample 用）

MINIO_COUNTS_KEY = "minio:counts"
MINIO_TRIM_LOCK_PREFIX = "minio:trim:lock:"
MINIO_TRIM_LOCK_TTL = 180
MINIO_RING_INDEX_KEY = "minio:ring:index"

_RELEASE_LOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


async def run_in_executor(executor: concurrent.futures.Executor, func: Callable, *args, **kwargs):
    loop = asyncio.get_running_loop()
    bound = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, bound)


async def run_in_io_executor(func: Callable, *args, **kwargs):
    return await run_in_executor(EXECUTOR_IO, func, *args, **kwargs)


class MinioManager:
    def __init__(
        self,
        redis_client: redis.Redis,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool,
        counts_key: str,
        ring_index_key: str,
        trim_lock_prefix: str,
        trim_lock_ttl: int,
        trim_batch: int,
        trim_max_delete: int,
        io_executor: concurrent.futures.Executor,
        upload_executor: concurrent.futures.Executor,
        trim_executor: concurrent.futures.Executor,
        ring_enabled: bool = False,
        monitor_callback: Callable[[str, str, float], None] | None = None,
    ) -> None:
        self._redis = redis_client
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.secure = secure
        self.counts_key = counts_key
        self.ring_index_key = ring_index_key
        self.trim_lock_prefix = trim_lock_prefix
        self.trim_lock_ttl = trim_lock_ttl
        self.trim_batch = trim_batch
        self.trim_max_delete = trim_max_delete
        self._io_executor = io_executor
        self._upload_executor = upload_executor
        self._trim_executor = trim_executor
        self._ring_enabled = ring_enabled
        self._monitor_callback = monitor_callback
        self._client: Minio | None = None
        self._release_lock_script = redis_client.register_script(_RELEASE_LOCK_LUA)

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    def initialize(self) -> None:
        try:
            client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
            if not client.bucket_exists(self.bucket):
                client.make_bucket(self.bucket)
            self._client = client
            print(f"[minio] ready -> {self.endpoint} bucket={self.bucket}")
        except Exception as e:
            self._client = None
            print(f"[minio][error] init failed: {e}")

    async def put_bytes(
        self, obj_key: str, data: bytes, content_type: str = "image/jpeg"
    ) -> str | None:
        if not self._client:
            return None
        bio = io.BytesIO(data)
        size = len(data)

        def _put():
            try:
                bio.seek(0)
                result = self._client.put_object(
                    self.bucket, obj_key, bio, size, content_type=content_type
                )
                return getattr(result, "etag", None)
            except S3Error as e:
                print(f"[minio][S3Error] put {obj_key}: {e}")
                return None
            except Exception as e:
                print(f"[minio][error] put {obj_key}: {e}")
                return None

        return await self._run_in_executor(self._upload_executor, _put)

    async def assign_ring_slot(self, safe_id: str, ring_size: int) -> Optional[int]:
        if ring_size <= 0:
            return None
        counter = await self._redis_hincrby(self.ring_index_key, safe_id, 1)
        if counter <= 0:
            return None
        return (counter - 1) % ring_size

    async def record_ring_success(self, safe_id: str, ring_size: int) -> None:
        if ring_size <= 0:
            await self.set_frame_count(safe_id, 0)
            return
        current = await self.get_frame_count(safe_id)
        if current < 0:
            await self.set_frame_count(safe_id, 0)
            current = 0
        if current < ring_size:
            await self.increment_frame_count(safe_id, 1)
        elif current > ring_size:
            await self.set_frame_count(safe_id, ring_size)

    async def trim_prefix(
        self, prefix: str, keep_last_n: int, safe_id: str | None = None
    ) -> int:
        if self._ring_enabled:
            return 0
        if not self._client or keep_last_n < 0:
            return 0

        safe_id = safe_id or prefix.rstrip("/").split("/")[0]
        lock_key = f"{self.trim_lock_prefix}{safe_id}"
        token = uuid4().hex
        acquired = await self._acquire_lock(lock_key, token)
        if not acquired:
            return 0

        deleted = 0
        renew_stop = asyncio.Event()
        renew_task: asyncio.Task | None = None

        if self.trim_lock_ttl > 0:
            interval = max(self.trim_lock_ttl / 3.0, 1.0)

            async def _renew_lock():
                try:
                    while True:
                        try:
                            await asyncio.wait_for(renew_stop.wait(), timeout=interval)
                            break
                        except asyncio.TimeoutError:
                            ok = await self._extend_lock(lock_key, token, self.trim_lock_ttl)
                            if not ok:
                                print(f"[redis][warn] extend lock failed {lock_key}")
                                break
                except asyncio.CancelledError:
                    raise

            renew_task = asyncio.create_task(_renew_lock())

        try:
            curr_count = await self.get_frame_count(safe_id)
            to_delete = max(curr_count - keep_last_n, 0)
            if to_delete <= 0:
                print(
                    f"[minio][trim][skip] prefix={prefix} curr={curr_count} keep={keep_last_n}"
                )
                return 0

            to_delete = min(to_delete, self.trim_max_delete)
            if to_delete <= 0:
                return 0

            def _trim_batch():
                deleted_total = 0
                visited = 0
                batch: list[DeleteObject] = []
                batch_limit = max(self.trim_batch, 1)
                try:
                    iterator = self._client.list_objects(
                        self.bucket, prefix=prefix, recursive=True
                    )
                    for obj in iterator:
                        if visited >= to_delete:
                            break
                        batch.append(DeleteObject(obj.object_name))
                        visited += 1
                        if len(batch) >= batch_limit or visited >= to_delete:
                            errors = self._client.remove_objects(self.bucket, batch)
                            failed = 0
                            for err in errors:
                                failed += 1
                                print(
                                    f"[minio][trim][warn] remove {getattr(err, 'object_name', '?')}: {getattr(err, 'message', err)}"
                                )
                            deleted_total += len(batch) - failed
                            batch = []
                    if batch and visited <= to_delete:
                        errors = self._client.remove_objects(self.bucket, batch)
                        failed = 0
                        for err in errors:
                            failed += 1
                            print(
                                f"[minio][trim][warn] remove {getattr(err, 'object_name', '?')}: {getattr(err, 'message', err)}"
                            )
                        deleted_total += len(batch) - failed
                    return deleted_total, visited
                except Exception as e:
                    print(f"[minio][trim][error] prefix={prefix} {e}")
                    return deleted_total, visited

            start_monotonic = time.monotonic()
            deleted, scanned = await self._run_in_executor(
                self._trim_executor, _trim_batch
            )
            elapsed_ms = (time.monotonic() - start_monotonic) * 1000.0
            if deleted > 0:
                await self.increment_frame_count(safe_id, -deleted)
            rest_est = max(curr_count - deleted, 0)
            print(
                f"[minio][trim] prefix={prefix} target={to_delete} scanned={scanned} deleted={deleted} rest_est={rest_est} keep={keep_last_n}"
            )
            if self._monitor_callback and deleted >= 0:
                self._monitor_callback(safe_id, "minio_trim", elapsed_ms)
            return deleted
        finally:
            renew_stop.set()
            if renew_task:
                try:
                    await renew_task
                except Exception as e:
                    print(f"[redis][warn] lock renew task error {lock_key}: {e}")
            await self._release_lock(lock_key, token)

    async def increment_frame_count(self, safe_id: str, delta: int) -> int:
        return await self._redis_hincrby(self.counts_key, safe_id, delta)

    async def get_frame_count(self, safe_id: str) -> int:
        return await self._redis_hget(self.counts_key, safe_id)

    async def set_frame_count(self, safe_id: str, value: int) -> bool:
        return await self._redis_hset(self.counts_key, safe_id, int(value))

    async def _redis_hincrby(self, key: str, field: str, delta: int) -> int:
        try:
            result = await self._run_in_executor(
                self._io_executor, self._redis.hincrby, key, field, delta
            )
            return int(result)
        except Exception as e:
            print(f"[redis][error] hincrby {key} {field} {delta}: {e}")
            return -1

    async def _redis_hget(self, key: str, field: str) -> int:
        try:
            value = await self._run_in_executor(
                self._io_executor, self._redis.hget, key, field
            )
            if value is None:
                return -1
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8")
                except Exception:
                    pass
            return int(value)
        except (TypeError, ValueError):
            return -1
        except Exception as e:
            print(f"[redis][error] hget {key} {field}: {e}")
            return -1

    async def _redis_hset(self, key: str, field: str, value: int) -> bool:
        try:
            await self._run_in_executor(
                self._io_executor, self._redis.hset, key, field, int(value)
            )
            return True
        except Exception as e:
            print(f"[redis][error] hset {key} {field} {value}: {e}")
            return False

    async def _acquire_lock(self, lock_key: str, token: str) -> bool:
        ttl = max(self.trim_lock_ttl, 1) if self.trim_lock_ttl > 0 else None
        try:
            result = await self._run_in_executor(
                self._io_executor,
                self._redis.set,
                lock_key,
                token,
                nx=True,
                ex=ttl,
            )
            return bool(result)
        except Exception as e:
            print(f"[redis][error] set {lock_key}: {e}")
            return False

    async def _release_lock(self, lock_key: str, token: str) -> bool:
        try:
            result = await self._run_in_executor(
                self._io_executor,
                self._release_lock_script,
                keys=[lock_key],
                args=[token],
            )
            return int(result or 0) > 0
        except Exception as e:
            print(f"[redis][error] release lock {lock_key}: {e}")
            return False

    async def _extend_lock(self, lock_key: str, token: str, ttl: int) -> bool:
        try:
            curr = await self._run_in_executor(self._io_executor, self._redis.get, lock_key)
            if curr != token:
                return False
            result = await self._run_in_executor(
                self._io_executor, self._redis.expire, lock_key, max(ttl, 1)
            )
            return bool(result)
        except Exception as e:
            print(f"[redis][error] expire {lock_key}: {e}")
            return False

    async def _run_in_executor(
        self, executor: concurrent.futures.Executor, func: Callable, *args, **kwargs
    ):
        loop = asyncio.get_running_loop()
        bound = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, bound)


MINIO_MONITOR_CALLBACK: Callable[[str, str, float], None] | None = (
    (lambda safe_id, stage, duration: PERFORMANCE_MONITOR.record_stage(safe_id, stage, duration))
    if MONITORING_ENABLED
    else None
)

MINIO_MANAGER = MinioManager(
    r,
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    bucket=MINIO_BUCKET,
    secure=MINIO_SECURE,
    counts_key=MINIO_COUNTS_KEY,
    ring_index_key=MINIO_RING_INDEX_KEY,
    trim_lock_prefix=MINIO_TRIM_LOCK_PREFIX,
    trim_lock_ttl=MINIO_TRIM_LOCK_TTL,
    trim_batch=MINIO_TRIM_BATCH,
    trim_max_delete=MINIO_TRIM_MAX_DELETE,
    io_executor=EXECUTOR_IO,
    upload_executor=MINIO_UPLOAD_EXECUTOR,
    trim_executor=MINIO_TRIM_EXECUTOR,
    ring_enabled=MINIO_RING_ENABLED,
    monitor_callback=MINIO_MONITOR_CALLBACK,
)

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


def build_ring_obj_key(safe_id: str, slot_index: int) -> str:
    tail = "_intrussion" if MINIO_USE_INTRUSSION_SUFFIX else ""
    return f"{safe_id}/ring/{slot_index:06d}{tail}.jpg"

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
        decode_ms = (t1 - t0) * 1000

        if frame is None:
            result = build_state_payload(
                StreamState.ERROR,
                decode_ms,
                **localized_message_payload("error.decoding_failed", field_name="error"),
            )
            return result, prev_frame

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 黑屏
        if float(grey.mean()) < float(current_threshold):
            result = build_state_payload(StreamState.BLACK_SCREEN, decode_ms)
            return result, grey

        # 2. 遮挡
        edges = cv2.Canny(grey, int(edge_params["low"]), int(edge_params["high"]))
        edge_ratio = float(cv2.countNonZero(edges)) / float(grey.size)
        if edge_ratio < float(edge_params["min_ratio"]):
            result = build_state_payload(
                StreamState.OCCLUDED,
                decode_ms,
                edge_ratio=round(edge_ratio * 100.0, 2),
            )
            return result, grey

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
                result = build_state_payload(
                    StreamState.TAMPERED,
                    decode_ms,
                    diff_mean=round(diff_mean, 2),
                    high_diff_ratio=round(high_diff_ratio, 2),
                    roi_ratio=round(roi_ratio, 2),
                )
                return result, grey

        # 4. 正常
        result = build_state_payload(
            StreamState.NORMAL,
            decode_ms,
            edge_ratio=round(edge_ratio * 100.0, 2),
        )
        return result, grey

    except Exception as e:
        print(f"[analyze][error] {e}")
        result = build_state_payload(StreamState.ERROR, 0.0, error=str(e))
        return result, prev_frame


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
    start_monotonic = time.monotonic()
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
                    if MONITORING_ENABLED:
                        elapsed_ms = (time.monotonic() - start_monotonic) * 1000.0
                        PERFORMANCE_MONITOR.record_stage(stream_name, "redis_fetch", elapsed_ms)
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

            if MONITORING_ENABLED:
                PERFORMANCE_MONITOR.record_frame(
                    stream_name,
                    {
                        "capture": cap_ms,
                        "queue": queue_ms,
                        "decode": decode_ms,
                        "analysis": analysis_ms,
                        "processing": proc_ms,
                        "end_to_end": e2e_ms,
                    },
                    state=analysis.get("state"),
                    ts_wall=meta.get("ts_wall_iso"),
                    ts3=ts3_iso,
                    ts4=ts4_iso,
                )

            # ========== 异常判定 ==========
            is_abnormal = analysis.get("state") not in ("Normal", None)

            # ========== MinIO: 仍然写“原始 JPEG”，不做叠加/再编码 ==========
            raw_jpeg = jpeg
            ts_ms = int(time.time() * 1000)
            obj_key = build_obj_key(stream_name, ts_ms)

            # 保存策略：若你只想异常时写 -> 环境变量设 MINIO_SAVE_MODE=abnormal
            save_this = (MINIO_SAVE_MODE == "all") or (MINIO_SAVE_MODE == "abnormal" and is_abnormal)
            _minio_pending_refs: List[dict] = []
            upload_durations: List[float] = []
            if save_this and MINIO_MANAGER.is_ready:
                safe_id = safe_filename(stream_name)
                primary_obj_key: Optional[str] = None
                primary_reason = "abnormal" if is_abnormal else "all"

                if MINIO_RING_ENABLED and MINIO_MAX_FRAMES_PER_STREAM > 0:
                    slot_index = await MINIO_MANAGER.assign_ring_slot(
                        safe_id, MINIO_MAX_FRAMES_PER_STREAM
                    )
                    if slot_index is not None:
                        ring_key = build_ring_obj_key(safe_id, slot_index)
                        t_upload_start = time.monotonic()
                        etag = await MINIO_MANAGER.put_bytes(
                            ring_key, raw_jpeg, content_type="image/jpeg"
                        )
                        if etag:
                            upload_ms = (time.monotonic() - t_upload_start) * 1000.0
                            upload_durations.append(upload_ms)
                            _minio_pending_refs.append(
                                {
                                    "type": "frame",
                                    "key": ring_key,
                                    "minio_object_key": ring_key,
                                    "etag": etag,
                                    "reason": primary_reason,
                                }
                            )
                            primary_obj_key = ring_key
                            await MINIO_MANAGER.record_ring_success(
                                safe_id, MINIO_MAX_FRAMES_PER_STREAM
                            )

                if primary_obj_key is None:
                    t_upload_start = time.monotonic()
                    etag = await MINIO_MANAGER.put_bytes(
                        obj_key, raw_jpeg, content_type="image/jpeg"
                    )
                    if etag:
                        upload_ms = (time.monotonic() - t_upload_start) * 1000.0
                        upload_durations.append(upload_ms)
                        _minio_pending_refs.append(
                            {
                                "type": "frame",
                                "key": obj_key,
                                "minio_object_key": obj_key,
                                "etag": etag,
                                "reason": primary_reason,
                            }
                        )
                        primary_obj_key = obj_key
                        if MINIO_RING_ENABLED and MINIO_MAX_FRAMES_PER_STREAM > 0:
                            await MINIO_MANAGER.record_ring_success(
                                safe_id, MINIO_MAX_FRAMES_PER_STREAM
                            )
                        else:
                            new_count = await MINIO_MANAGER.increment_frame_count(
                                safe_id, 1
                            )
                            if (
                                MINIO_TRIM_INTERVAL > 0
                                and new_count > 0
                                and new_count % MINIO_TRIM_INTERVAL == 0
                            ):

                                async def _trim_task():
                                    try:
                                        await MINIO_MANAGER.trim_prefix(
                                            prefix=f"{safe_id}/",
                                            keep_last_n=MINIO_MAX_FRAMES_PER_STREAM,
                                            safe_id=safe_id,
                                        )
                                    except Exception as e:
                                        print(f"[minio][trim][error] task {safe_id}: {e}")

                                asyncio.create_task(_trim_task())

                if MONITORING_ENABLED and upload_durations:
                    PERFORMANCE_MONITOR.record_stage(
                        stream_name, "minio_upload", sum(upload_durations)
                    )

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
            if _minio_pending_refs:
                payload.setdefault("obj_refs", []).extend(_minio_pending_refs)

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


@router.get("/metrics/performance", summary="获取性能统计")
async def get_performance_metrics():
    if not MONITORING_ENABLED:
        return {"enabled": False}
    snapshot = PERFORMANCE_MONITOR.snapshot()
    snapshot["enabled"] = True
    return snapshot


@router.get("/metrics/performance/{stream_name}", summary="获取某路视频流的性能统计")
async def get_stream_performance_metrics(stream_name: str):
    if not MONITORING_ENABLED:
        return {"enabled": False, "stream": stream_name, "stages": {}, "frames": []}
    snapshot = PERFORMANCE_MONITOR.snapshot(stream=stream_name)
    snapshot["enabled"] = True
    return snapshot


@router.delete("/metrics/performance", summary="重置性能统计")
async def reset_performance_metrics():
    if MONITORING_ENABLED:
        PERFORMANCE_MONITOR.reset()
    return {"enabled": MONITORING_ENABLED}


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
        return {"ok": True, **localized_message_payload("service.already_started")}
    SERVICE_ACTIVE = True
    print("[service] started")
    return {"ok": True, **localized_message_payload("service.started")}


@router.post("/stop")
async def stop_service():
    global SERVICE_ACTIVE, STREAM_WORKERS, SUBSCRIBED_STREAMS, PREV_FRAMES
    if not SERVICE_ACTIVE:
        print("[service] stop called, but already inactive")
        return {"ok": True, **localized_message_payload("service.already_stopped")}

    SERVICE_ACTIVE = False
    print("[service] stopping workers...")
    for name, task in list(STREAM_WORKERS.items()):
        print(f"[service] cancelling worker {name}")
        task.cancel()
    STREAM_WORKERS.clear()
    SUBSCRIBED_STREAMS.clear()
    PREV_FRAMES.clear()
    print("[service] stopped, resources released")
    return {"ok": True, **localized_message_payload("service.stopped")}


@router.post("/subscribe")
async def subscribe(body: Dict = Body(...)):
    global CURRENT_THRESHOLD, EDGE_PARAMS, TAMPER_PARAMS

    if not SERVICE_ACTIVE:
        print("[subscribe] rejected, service not started")
        return make_error_response("error.service_not_started", status_code=400)

    streams = body.get("streams", [])
    if not isinstance(streams, list):
        return make_error_response("error.streams_not_list", status_code=400)

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
    MINIO_MANAGER.initialize()


# 挂载 router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5020, log_level="info")
