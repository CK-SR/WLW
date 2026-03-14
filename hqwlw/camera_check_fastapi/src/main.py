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
from minio.error import S3Error

try:
    from minio.commonconfig import CopySource  # ← 新增：copy_object 需要
except Exception:
    CopySource = None

try:
    from minio.lifecycleconfig import Expiration, Filter, LifecycleConfig, Rule
except ImportError:  # pragma: no cover - 兼容旧版 SDK
    Expiration = Filter = LifecycleConfig = Rule = None  # type: ignore


EXECUTOR_CPU = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(8,os.cpu_count()*2)
)

EXECUTOR_IO = concurrent.futures.ThreadPoolExecutor(max_workers=64)
MINIO_UPLOAD_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=32)

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

# 正在做缩环迁移的流；迁移期间该流的帧“可分析但不上传 MinIO”
MINIO_MIGRATING: Set[str] = set()


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

SERVICE_RESTARTED = True  # 初始化标志，默认认为服务重启

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
MINIO_MAX_FRAMES_PER_STREAM = int(config_settings.minio.max_frames_per_stream)

# 是否在文件名加 intrussion 后缀；仅命名习惯，无功能差异
MINIO_USE_INTRUSSION_SUFFIX = bool(config_settings.minio.use_intrussion_suffix)

# 保存策略：all / abnormal / none
MINIO_SAVE_MODE = config_settings.minio.save_mode

WS_SEND_MODE = config_settings.stream.ws_send_mode  # all / abnormal

# === per-stream 配置与冷静期 ===
from typing import DefaultDict

# 每路自定义最大帧数上限；未设置则回退到 MINIO_MAX_FRAMES_PER_STREAM
PER_STREAM_MAX_FRAMES: Dict[str, int] = {}

# 每路冷静期（秒）；未设置则为 0
PER_STREAM_COOLDOWN: Dict[str, int] = {}

# 记录每路“最近一次检测到异常”的时间戳（epoch 秒）
LAST_INTRUSION_TS: Dict[str, float] = {}

# 当订阅把 minio_storage_per_stream 调小后，标记该流需要做一次“老对象裁剪”
STREAMS_NEED_TRIM: Set[str] = set()


def get_max_frames(stream_name: str) -> int:
    return int(PER_STREAM_MAX_FRAMES.get(stream_name, MINIO_MAX_FRAMES_PER_STREAM))

def get_cooldown_seconds(stream_name: str) -> int:
    return int(PER_STREAM_COOLDOWN.get(stream_name, 0))

def _resolve_minio_lifecycle_days() -> int:
    if hasattr(config_settings.minio, "lifecycle_days"):
        value = getattr(config_settings.minio, "lifecycle_days")
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    env_value = os.getenv("MINIO_LIFECYCLE_DAYS")
    if env_value not in (None, ""):
        try:
            return int(env_value)
        except ValueError:
            pass
    return 3


MINIO_LIFECYCLE_DAYS = _resolve_minio_lifecycle_days()

# 获取当前MinIO的对象数量来决定是否缩环
async def get_current_minio_capacity(safe_id: str) -> int:
    try:
        count = await MINIO_MANAGER.ring_object_count(safe_id)
        return count
    except Exception as e:
        print(f"[minio][error] failed to get object count for {safe_id}: {e}")
        return 0  # 如果失败则认为容量为0


async def run_in_executor(executor: concurrent.futures.Executor, func: Callable, *args, **kwargs):
    loop = asyncio.get_running_loop()
    bound = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, bound)


async def run_in_io_executor(func: Callable, *args, **kwargs):
    return await run_in_executor(EXECUTOR_IO, func, *args, **kwargs)


class MinioManager:
    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool,
        lifecycle_days: int,
        io_executor: concurrent.futures.Executor,
        upload_executor: concurrent.futures.Executor,
        monitor_callback: Callable[[str, str, float], None] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.secure = secure
        self.lifecycle_days = lifecycle_days
        self._io_executor = io_executor
        self._upload_executor = upload_executor
        self._monitor_callback = monitor_callback
        self._client: Minio | None = None
        self._ring_ptr: Dict[str, int] = {}
        self._ring_lock = asyncio.Lock()

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
            self._apply_lifecycle_policy()
            # ★★★ 新增：确保公共读取策略
            self._ensure_public_read()
        except Exception as e:
            self._client = None
            print(f"[minio][error] init failed: {e}")

    def _apply_lifecycle_policy(self) -> None:
        if not self._client:
            return

        if self.lifecycle_days <= 0:
            self._clear_lifecycle_policy()
            return

        days = int(self.lifecycle_days)

        # 1) 优先：对象 API（新 SDK）
        try:
            from minio.lifecycleconfig import LifecycleConfig, Rule, Expiration
            try:
                # 有些版本在 lifecycleconfig 里就有 Filter
                from minio.lifecycleconfig import Filter as _Filter
            except ImportError:
                # 另一些版本在 commonconfig 里
                from minio.commonconfig import Filter as _Filter

            # 注意：新版本参数名是 rule_filter
            rule = Rule(
                rule_id="expire-all",
                status="Enabled",
                rule_filter=_Filter(prefix=""),
                expiration=Expiration(days=days),
            )
            config = LifecycleConfig([rule])
            self._client.set_bucket_lifecycle(self.bucket, config)
            print(f"[minio] lifecycle configured (object API): expire after {days} days")
            return

        except Exception as e_obj:
            # 2) 旧 SDK Fallback：XML
            xml = (
                '<LifecycleConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                "<Rule><ID>expire-all</ID><Status>Enabled</Status>"
                "<Filter><Prefix></Prefix></Filter>"
                f"<Expiration><Days>{days}</Days></Expiration>"
                "</Rule></LifecycleConfiguration>"
            )
            try:
                self._client.set_bucket_lifecycle(self.bucket, xml)
                print(f"[minio] lifecycle configured (XML fallback): expire after {days} days")
            except Exception as e_xml:
                print(f"[minio][warn] set lifecycle failed: {e_obj} | xml_fallback_error: {e_xml}")

    def _clear_lifecycle_policy(self) -> None:
        """Remove bucket lifecycle configuration when lifecycle_days <= 0."""

        if not self._client:
            return

        # 1) SDK API 优先
        try:
            self._client.delete_bucket_lifecycle(self.bucket)
            print("[minio] lifecycle disabled: existing lifecycle policy removed")
            return
        except Exception as e_delete:
            # 2) 兼容旧版本/不同命名实现
            remove_fn = getattr(self._client, "remove_bucket_lifecycle", None)
            if callable(remove_fn):
                try:
                    remove_fn(self.bucket)
                    print("[minio] lifecycle disabled: existing lifecycle policy removed")
                    return
                except Exception as e_remove:
                    print(
                        f"[minio][warn] clear lifecycle failed: "
                        f"delete_error={e_delete} | remove_error={e_remove}"
                    )
                    return

            print(f"[minio][warn] clear lifecycle failed: {e_delete}")


    def _ensure_public_read(self) -> None:
        """
        为当前 bucket 设置只读的公共访问策略（允许匿名 GetObject）。
        失败时仅告警，不中断流程。
        """
        client = self._client
        bucket = self.bucket
        if not client or not bucket:
            return

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket}/*"],
                }
            ],
        }
        try:
            client.set_bucket_policy(bucket, json.dumps(policy))
            print(f"[minio] bucket={bucket} policy=public-read applied")
        except Exception as exc:
            print(f"[minio][warn] set public-read policy failed: {exc}")


    async def warmup_ring_ptr(self, safe_id: str, ring_size: int) -> None:
        """
        在服务重启后，根据 MinIO 现有对象恢复“最新槽位号”，使下一次写入从最新之后开始。
        规则：
          1) 优先从 Redis 读取已持久化的指针（如果存在且 < ring_size）
          2) 否则列举 {safe_id}/ring/ 前缀下所有对象，按 last_modified 找到最新对象，
             再从 object_key 中解析槽位号，设置为“当前指针”（下一次写入 +1）。
        """
        if ring_size <= 0:
            return
        # 1) 先看 Redis 是否有持久化指针
        try:
            persisted = r.hget("minio:ring_ptr", safe_id)
            if persisted is not None:
                pi = int(persisted)
                if 0 <= pi < ring_size:
                    self._ring_ptr[safe_id] = pi
                    return
        except Exception:
            pass

        # 2) 扫描 MinIO 对象，找到最新的那一个
        if not self._client:
            return

        prefix = f"{safe_id}/ring/"
        def _list_all():
            # 列举所有对象；MinIO SDK 的 list_objects 是生成器
            return list(self._client.list_objects(self.bucket, prefix=prefix, recursive=True))

        items = await self._run_in_executor(self._io_executor, _list_all)
        if not items:
            self._ring_ptr[safe_id] = -1
            return

        # 找 last_modified 最新的对象
        latest = max(items, key=lambda o: getattr(o, "last_modified", None) or 0)
        key = latest.object_name
        # 解析槽位号：.../ring/000123[_intrussion].jpg
        m = re.search(r"/ring/(\d{6})", key)
        if not m:
            self._ring_ptr[safe_id] = -1
            return
        idx = int(m.group(1))
        self._ring_ptr[safe_id] = idx
        # 同步保存到 Redis，便于下次快速恢复
        try:
            r.hset("minio:ring_ptr", safe_id, idx)
        except Exception:
            pass

    

    
    
    async def ring_object_count(self, safe_id: str, at_least: Optional[int] = None) -> int:
        """
        统计 {safe_id}/ring/ 前缀下对象数量。
        - at_least: 若设置，则在计数达到该阈值时就提前返回（早停），降低列举成本。
        """
        if not self._client:
            return 0

        prefix = f"{safe_id}/ring/"

        def _count():
            c = 0
            for _ in self._client.list_objects(self.bucket, prefix=prefix, recursive=True):
                c += 1
                if at_least is not None and c >= at_least:
                    return c
            return c

        return await self._run_in_executor(self._io_executor, _count)

    
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

    

    async def shrink_keep_last_n(self, safe_id: str, old_M: int, new_N: int) -> None:
        """
        将“缩环时刻最近的 new_N 帧”平移到 0..new_N-1（ring/000000..），删除其他对象，
        并把指针设置为 new_N-1（下一帧从 0 覆盖写）。
        需要：old_M >= new_N >= 0
        """
        if not self._client or new_N < 0 or old_M < 0 or new_N > old_M:
            return

        # N=0 => 清空并停写
        if new_N == 0:
            prefix = f"{safe_id}/ring/"
            def _delete_all():
                for o in self._client.list_objects(self.bucket, prefix=prefix, recursive=True):
                    try:
                        self._client.remove_object(self.bucket, o.object_name)
                    except Exception as e:
                        print(f"[minio][warn] remove {o.object_name} failed: {e}")
            await self._run_in_executor(self._io_executor, _delete_all)
            # 指针置 -1
            self._ring_ptr[safe_id] = -1
            try:
                r.hset("minio:ring_ptr", safe_id, -1)
            except Exception:
                pass
            return

        # 取指针 P（上一帧写到的槽位）。没有就先 warmup
        if safe_id not in self._ring_ptr:
            await self.warmup_ring_ptr(safe_id, old_M)
        P = int(self._ring_ptr.get(safe_id, -1))
        if P < 0:  # 没有历史帧，直接结束：只需要把指针设为 -1 让下次从 0 开写
            try:
                r.hset("minio:ring_ptr", safe_id, -1)
            except Exception:
                pass
            self._ring_ptr[safe_id] = -1
            return

        M = int(old_M)
        N = int(new_N)

        # 计算最近 N 帧的连续区间 S（从旧到新）
        s = (P - (N - 1) + M) % M
        e = P
        if s <= e:
            S_slots = list(range(s, e + 1))                # 不跨0
        else:
            S_slots = list(range(s, M)) + list(range(0, e + 1))  # 跨0

        # 免复制特例：S 恰好是 0..N-1（即 P == N-1 且不跨0）
        if s == 0 and e == N - 1:
            # 仅删除 N..M-1
            prefix = f"{safe_id}/ring/"
            def _delete_high_tail():
                for o in self._client.list_objects(self.bucket, prefix=prefix, recursive=True):
                    # .../ring/000123*.jpg 解析槽位
                    m = re.search(r"/ring/(\d+)", o.object_name)
                    if not m: 
                        continue
                    idx = int(m.group(1))
                    if idx >= N:
                        try:
                            self._client.remove_object(self.bucket, o.object_name)
                        except Exception as ee:
                            print(f"[minio][warn] remove {o.object_name} failed: {ee}")
            await self._run_in_executor(self._io_executor, _delete_high_tail)

            # 指针校正为 N-1（下一帧写 0）
            self._ring_ptr[safe_id] = N - 1
            try:
                r.hset("minio:ring_ptr", safe_id, N - 1)
            except Exception:
                pass
            return

        # 一般情形：需要平移（server-side copy），然后清理
        # 目标键集合：dst(i) = 0..N-1
        dst_keys = [f"{safe_id}/ring/{i:06d}{'_intrussion' if MINIO_USE_INTRUSSION_SUFFIX else ''}.jpg" for i in range(N)]

        def _copy_and_cleanup():
            # 1) 平移：S[i] -> i
            for i, slot in enumerate(S_slots):
                src_key = f"{safe_id}/ring/{slot:06d}{'_intrussion' if MINIO_USE_INTRUSSION_SUFFIX else ''}.jpg"
                dst_key = dst_keys[i]
                if src_key == dst_key:
                    continue
                try:
                    if CopySource is None:
                        # SDK 极老版本兜底：下载再上传（不建议，但保证可用）
                        data = self._client.get_object(self.bucket, src_key).read()
                        bio = io.BytesIO(data)
                        self._client.put_object(self.bucket, dst_key, bio, len(data), content_type="image/jpeg")
                    else:
                        self._client.copy_object(self.bucket, dst_key, CopySource(self.bucket, src_key))
                except Exception as ee:
                    print(f"[minio][warn] copy {src_key} -> {dst_key} failed: {ee}")

            # 2) 删除除 dst(0..N-1) 之外的所有对象
            keep = set(dst_keys)
            prefix = f"{safe_id}/ring/"
            for o in self._client.list_objects(self.bucket, prefix=prefix, recursive=True):
                if o.object_name not in keep:
                    try:
                        self._client.remove_object(self.bucket, o.object_name)
                    except Exception as ee:
                        print(f"[minio][warn] remove {o.object_name} failed: {ee}")

        # 与 next_ring_slot 的自增串行化，降低竞态
        async with self._ring_lock:
            await self._run_in_executor(self._io_executor, _copy_and_cleanup)
            # 3) 指针设为 N-1（下一帧写 0）
            self._ring_ptr[safe_id] = N - 1
            try:
                r.hset("minio:ring_ptr", safe_id, N - 1)
            except Exception:
                pass

    async def next_ring_slot(self, safe_id: str, ring_size: int) -> Optional[int]:
        if ring_size <= 0:
            return None
        async with self._ring_lock:
            # 如果当前没有指针，尝试 Redis/MinIO 恢复
            if safe_id not in self._ring_ptr:
                await self.warmup_ring_ptr(safe_id, ring_size)

            current = self._ring_ptr.get(safe_id, -1)
            next_index = (current + 1) % ring_size
            self._ring_ptr[safe_id] = next_index
            # 持久化到 Redis
            try:
                r.hset("minio:ring_ptr", safe_id, next_index)
            except Exception:
                pass
            return next_index

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
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    bucket=MINIO_BUCKET,
    secure=MINIO_SECURE,
    lifecycle_days=MINIO_LIFECYCLE_DAYS,
    io_executor=EXECUTOR_IO,
    upload_executor=MINIO_UPLOAD_EXECUTOR,
    monitor_callback=MINIO_MONITOR_CALLBACK,
)

# ★ 安全文件名生成器
def safe_filename(stream_name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', "_", stream_name)


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

def annotate_image_with_results(image: bytes, analysis: dict) -> bytes:
    # Decode the image
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return image

    # Example: Adding a simple annotation on the image
    text = analysis.get("state", "Unknown")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green color for the text
    thickness = 2
    position = (50, 50)  # Position for the text

    # Overlay text on the image
    cv2.putText(img, text, position, font, font_scale, color, thickness)

    # Re-encode the image back to JPEG format
    _, jpeg_img = cv2.imencode('.jpg', img)
    return jpeg_img.tobytes()

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


async def stream_worker(stream_name: str, annotate_images: bool):
    print(f"[worker start] {stream_name}")

    video_writer = None
    frame_count = 0

    try:
        while True:
            # ========== 获取帧 ==========
            meta, jpeg = await get_one_frame(stream_name, timeout_sec=1.0)
            if jpeg is None:
                continue

            safe_id = safe_filename(stream_name)  # ← 新增：后面上传或裁剪都会用到
            ts3_epoch = time.time()
            ts3_iso = now_iso_utc()

            # ========== 冷静期判断 & 调用分析 ==========
            prev_frame = PREV_FRAMES.get(stream_name)
            t_proc_start = time.monotonic()

            cooldown_s = get_cooldown_seconds(stream_name)
            now_epoch = time.time()
            in_cooldown = False
            if cooldown_s > 0:
                last_intr = LAST_INTRUSION_TS.get(stream_name, 0.0)
                if last_intr > 0 and (now_epoch - last_intr) < cooldown_s:
                    in_cooldown = True

            if in_cooldown:
                # 冷静期：跳过耗时的算法检测与解码；直接视为“正常”，并标记 in_cooldown
                analysis = build_state_payload(StreamState.NORMAL, decode_ms=None, in_cooldown=True,
                                              cooldown_left=round(cooldown_s - (now_epoch - last_intr), 2))
                curr_grey = None
            else:
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
            if is_abnormal:
                LAST_INTRUSION_TS[stream_name] = time.time()

            # 这里的 annotate_images 决定是否需要对图像进行标注
            if annotate_images:
                jpeg = annotate_image_with_results(jpeg, analysis)

            # ========== 保存到 MinIO 逻辑 ==========
            # 这里继续上传 JPEG（原始图像或已标注图像）
            raw_jpeg = jpeg

            # 保存策略：若你只想异常时写 -> 环境变量设 MINIO_SAVE_MODE=abnormal
            save_this = (MINIO_SAVE_MODE == "all") or (MINIO_SAVE_MODE == "abnormal" and is_abnormal)
            _minio_pending_refs: List[dict] = []
            upload_durations: List[float] = []
            saved_obj_key = None  # 用于 WS 返回
            # 只在不处于缩环迁移时才写 MinIO
            if (
                save_this
                and MINIO_MANAGER.is_ready
                and get_max_frames(stream_name) > 0
                and safe_id not in MINIO_MIGRATING
            ):
                safe_id = safe_filename(stream_name)
                primary_reason = "abnormal" if is_abnormal else "all"

                # 按每路上限获取槽位（内部会自动从 Redis/MinIO 恢复指针）
                slot_index = await MINIO_MANAGER.next_ring_slot(
                    safe_id, get_max_frames(stream_name)
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
                        saved_obj_key = ring_key

            # === 与保存解耦：即使这帧没上传，也要按需裁剪一次 ===
            if stream_name in STREAMS_NEED_TRIM:
                try:
                    
                    await MINIO_MANAGER.enforce_quota(safe_id, get_max_frames(stream_name))
                finally:
                    STREAMS_NEED_TRIM.discard(stream_name)

            if MONITORING_ENABLED and upload_durations:
                PERFORMANCE_MONITOR.record_stage(
                    stream_name, "minio_upload", sum(upload_durations)
                )

            # ========== 组织 WS payload（仍包含分析结果）==========
            payload = {
                "stream": str(stream_name),
                "ts_wall": meta.get("ts_wall_iso"),
                "shape": meta.get("shape"),
                #"minio_bucket": MINIO_BUCKET,   # ★ 新增：WS 中返回 bucket 信息
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
                    },
                    "image_minio": f"http://{config_settings.minio.endpoint}/{MINIO_BUCKET}/{saved_obj_key}" if saved_obj_key else None
                },
            }


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
    global SERVICE_ACTIVE, CURRENT_THRESHOLD, EDGE_PARAMS, TAMPER_PARAMS, SERVICE_RESTARTED

    # 检查是否包含 `minio_storage_per_stream`，如果没有则返回 400 错误
    if "minio_storage_per_stream" not in body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="minio_storage_per_stream is required"
        )

    # Check for the 'annotate_images' flag and default it to False if not provided
    annotate_images = body.get("annotate_images", False)

    # ★ 不再拒绝：若未激活，则在订阅时自动启动
    if not SERVICE_ACTIVE:
        SERVICE_ACTIVE = True
        print("[service] auto-start by subscribe")

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
            STREAM_WORKERS[name] = asyncio.create_task(stream_worker(name, annotate_images))

    # === 新增：解析 per-stream 上限与冷静期 ===
    raw_limit = body.get("minio_storage_per_stream", 100000)
    raw_cooldown = body.get("intrusion_cooldown_seconds", None)

    # 允许两种形式：
    #  1) 整数：对所有订阅流生效
    #  2) dict：{stream_name: int}
    def _as_int(v, default=None):
        try:
            return int(v)
        except Exception:
            return default

    # 记录哪些流上限变小了，用于触发一次性裁剪
    decreased_streams: List[str] = []

    for name in SUBSCRIBED_STREAMS:
        # --- 上限 ---
        if SERVICE_RESTARTED:
            # 如果服务重启，则根据当前MinIO容量来设置 prev
            prev = await get_current_minio_capacity(safe_filename(name))
        else:
            # 服务未重启时，使用上一次POST请求的最大帧数
            prev = get_max_frames(name)
        if isinstance(raw_limit, dict):
            new_limit = _as_int(raw_limit.get(name), prev)
        elif raw_limit is not None:
            new_limit = _as_int(raw_limit, prev)
        else:
            new_limit = prev  # 未提供则保持

        PER_STREAM_MAX_FRAMES[name] = new_limit

        if MINIO_MANAGER.is_ready and new_limit > 0:
            safe_id = safe_filename(name)
            try:
                # 只有当 new_limit < prev 时才需要缩环迁移
                if new_limit < prev:
                    print(f"[subscribe] shrink {name}: M={prev} -> N={new_limit}")
                    async def _do_shrink(safe_id,prev,new_limit):
                        MINIO_MIGRATING.add(safe_id)
                        try:
                            await MINIO_MANAGER.shrink_keep_last_n(safe_id, old_M=prev, new_N=new_limit)
                        finally:
                            MINIO_MIGRATING.discard(safe_id)
                    asyncio.create_task(_do_shrink(safe_id,prev,new_limit))
                    
                else:
                    # 扩容或不变：无需处理。可按需清理高槽位历史重复（通常不需要）
                    pass
            except Exception as e:
                print(f"[subscribe][warn] shrink_keep_last_n failed for {name}: {e}")

        
            

        # --- 冷静期 ---
        if isinstance(raw_cooldown, dict):
            PER_STREAM_COOLDOWN[name] = _as_int(raw_cooldown.get(name), get_cooldown_seconds(name)) or 0
        elif raw_cooldown is not None:
            PER_STREAM_COOLDOWN[name] = _as_int(raw_cooldown, get_cooldown_seconds(name)) or 0
        else:
            # 未提供不变；首次订阅无值则为 0
            PER_STREAM_COOLDOWN.setdefault(name, 0)

    if decreased_streams:
        print(f"[subscribe] decreased limits for: {decreased_streams} -> will trim oldest objects once")

    print(f"[subscribe] epoch={curr_epoch}, subscribed={list(SUBSCRIBED_STREAMS)}")
    print(f"[params] current_threshold={CURRENT_THRESHOLD}, edge_params={EDGE_PARAMS}, tamper_params={TAMPER_PARAMS}")

    SERVICE_RESTARTED = False
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
