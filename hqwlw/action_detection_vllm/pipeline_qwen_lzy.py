from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict, deque
import asyncio
import threading
import queue


import cv2
import numpy as np
import torch
from PIL import Image
import redis
from configs.settings import settings as config_settings

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

from camera_check_fastapi.src.main import MinioManager, safe_filename


DEFAULT_ACTION_CLASSES: Dict[int, str] = {
    -1: "unknown",
    0: "smoking",
    1: "calling",
    2: "climb",
    3: "jump",
    4: "run",
    5: "takephoto",
    6: "walk",
    7: "carry",   # 原 carry → theft
}

# === 新增：哪些行为视为“正常行为”，不会算异常 ===
#NORMAL_CLASSES = {"walk", "walking"}  # walk / walking 都当作正常
UNKNOWN_CLASSES = {"unknown"}

# 其他一律算异常的话可以不用这个集合；如果想精确控制，可以改成：
ABNORMAL_CLASSES = {
    "walk",
    "smoking",
    "calling",
    "climb",
    "jump",
    "run",
    "takephoto",
    "carry",
}


# 允许的 id-name 映射（用于一致性校验）
VALID_ID_TO_NAME: Dict[int, str] = {
    -1: "unknown",
     0: "smoking",
     1: "calling",
     2: "climb",
     3: "jump",
     4: "run",
     5: "takephoto",
     6: "walk",
     7: "carry",
}
VALID_NAME_TO_ID: Dict[str, int] = {v: k for k, v in VALID_ID_TO_NAME.items()}

# === 异常告警的最小置信度阈值（宁可漏检）===
ABNORMAL_MIN_CONFIDENCE = float(os.environ.get("ABNORMAL_MIN_CONFIDENCE", "0.8"))

PRIMARY_MINIO_BUCKET = str(getattr(config_settings.minio, "bucket", "") or "")


def _resolve_action_bucket() -> str:
    # bucket = getattr(config_settings.minio, "action_bucket", None)
    # if not bucket:
    #     bucket = f"{PRIMARY_MINIO_BUCKET}-action" if PRIMARY_MINIO_BUCKET else "action-detection"
    # if bucket == PRIMARY_MINIO_BUCKET:
    #     bucket = f"{bucket}-ad"
    # return bucket
    return "action-detection-vllm"


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


MINIO_RING_ENABLED = bool(getattr(config_settings.minio, "ring_enabled", True))
MINIO_RING_SIZE = int(getattr(config_settings.minio, "max_frames_per_stream", 10000000))
MINIO_JPEG_QUALITY = int(getattr(config_settings.minio, "jpeg_quality", 85))
#DEFAULT_UPLOAD_ALL_FRAMES = str(getattr(config_settings.minio, "save_mode", "all")).lower() == "all"
DEFAULT_UPLOAD_ALL_FRAMES = False

# === 新增：冷静期（秒），默认 30s，环境变量可覆盖 ===
DEFAULT_COOLDOWN_SECONDS = float(os.environ.get("PIPELINE_COOLDOWN_SECONDS", "3"))

class UploadPreferences:
    def __init__(self) -> None:
        self.upload_all_frames: bool = DEFAULT_UPLOAD_ALL_FRAMES
        self.upload_annotated: bool = True
        self._lock = threading.Lock()

    def set_upload_all_frames(self, enabled: bool) -> None:
        with self._lock:
            self.upload_all_frames = bool(enabled)

    def set_upload_annotated(self, enabled: bool) -> None:
        with self._lock:
            self.upload_annotated = bool(enabled)

    def snapshot(self) -> Dict[str, bool]:
        with self._lock:
            return {
                "upload_all_frames": self.upload_all_frames,
                "upload_annotated": self.upload_annotated,
            }


UPLOAD_PREFS = UploadPreferences()


def set_upload_all_frames(enabled: bool) -> Dict[str, bool]:
    UPLOAD_PREFS.set_upload_all_frames(enabled)
    return UPLOAD_PREFS.snapshot()


def set_upload_annotated(enabled: bool) -> Dict[str, bool]:
    UPLOAD_PREFS.set_upload_annotated(enabled)
    return UPLOAD_PREFS.snapshot()


def get_upload_preferences() -> Dict[str, bool]:
    return UPLOAD_PREFS.snapshot()


def _draw_one_detection(frame_bgr: np.ndarray, bbox_xyxy: Tuple[int,int,int,int], label: str) -> np.ndarray:
    img = frame_bgr.copy()
    x1,y1,x2,y2 = bbox_xyxy
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, label, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return img


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



class BenchmarkRecorder:
    """
    简单的 benchmark 记录器：
    - record(stage, duration_ms): 记录某阶段一次耗时
    - record_drop(stage): 记录丢弃事件（只统计次数）
    - snapshot(): 返回 {stage: {count, avg_ms, max_ms}}
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, float]] = {}

    def record(self, stage: str, duration_ms: Optional[float]) -> None:
        if duration_ms is None:
            return
        with self._lock:
            s = self._stats.setdefault(stage, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
            s["count"] += 1
            s["total_ms"] += float(duration_ms)
            s["max_ms"] = max(s["max_ms"], float(duration_ms))

    def record_drop(self, stage: str) -> None:
        with self._lock:
            s = self._stats.setdefault(stage, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
            s["count"] += 1

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            out: Dict[str, Dict[str, float]] = {}
            for stage, s in self._stats.items():
                cnt = s["count"]
                avg = s["total_ms"] / cnt if cnt > 0 else 0.0
                out[stage] = {
                    "count": cnt,
                    "avg_ms": avg,
                    "max_ms": s["max_ms"],
                }
            return out



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
ACTION_MINIO_UPLOAD_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=32)

ACTION_MINIO_ENDPOINT = config_settings.minio.endpoint
ACTION_MINIO_ACCESS_KEY = config_settings.minio.access_key
ACTION_MINIO_SECRET_KEY = config_settings.minio.secret_key
ACTION_MINIO_BUCKET = _resolve_action_bucket()
ACTION_MINIO_SECURE = bool(config_settings.minio.secure)
ACTION_MINIO_LIFECYCLE_DAYS = 0



class ActionMinioManager(MinioManager):
    """Dedicated MinIO manager instance for action detection uploads."""

    def __init__(self, *args, **kwargs) -> None:
        # 使用线程锁来维护环形索引，避免基类中的 asyncio.Lock 绑定到不同事件循环时
        # 引发 RuntimeError。
        import threading
        import re

        self._ring_lock_sync = threading.Lock()
        self._warmup_done = set()          # 记录哪些 safe_id 已 warmup，避免重复扫 MinIO
        self._slot_re = re.compile(r"/ring/(\d{6})_")  # 支持 000123_raw.jpg / 000123_annotated.jpg
        super().__init__(*args, **kwargs)

    async def warmup_ring_ptr_from_minio(self, safe_id: str, ring_size: int) -> None:
        """
        仅依赖 MinIO 扫描 {safe_id}/ring/ 下的对象，恢复“最后写入槽位”指针。
        - 不依赖 Redis
        - 以 last_modified 最新为准；若 last_modified 相同，使用 object_name 做 tie-breaker
        - 扫不到则置 -1
        """
        if ring_size <= 0:
            return
        if not getattr(self, "_client", None):
            # MinIO 未初始化
            return

        # 避免频繁扫描：每个 safe_id 仅 warmup 一次
        with self._ring_lock_sync:
            if safe_id in self._warmup_done:
                return
            # 先标记，避免并发重复扫（即使扫失败也不会无限扫；你也可以改成失败不标记）
            self._warmup_done.add(safe_id)

        prefix = f"{safe_id}/ring/"

        def _list_all():
            # list_objects 返回生成器
            return list(self._client.list_objects(self.bucket, prefix=prefix, recursive=True))

        try:
            items = await self._run_in_executor(self._io_executor, _list_all)
        except Exception as e:
            print(f"[minio][warmup][warn] list_objects failed for {safe_id}: {e}")
            with self._ring_lock_sync:
                self._ring_ptr[safe_id] = -1
            return

        if not items:
            with self._ring_lock_sync:
                self._ring_ptr[safe_id] = -1
            return

        # last_modified 最新优先；相同时间用 object_name 做稳定排序
        def _key(o):
            lm = getattr(o, "last_modified", None)
            # lm 可能是 datetime；None 兜底为 0
            return (lm or 0, getattr(o, "object_name", ""))

        latest = max(items, key=_key)
        obj_name = getattr(latest, "object_name", "")

        m = self._slot_re.search(obj_name)
        if not m:
            # 兜底：尝试匹配没有下划线的情况：.../ring/000123.jpg（如果你未来改名）
            import re as _re
            m2 = _re.search(r"/ring/(\d{6})", obj_name)
            if not m2:
                with self._ring_lock_sync:
                    self._ring_ptr[safe_id] = -1
                return
            slot = int(m2.group(1))
        else:
            slot = int(m.group(1))

        # 防御：slot 可能来自旧 ring_size（你改过 max_frames_per_stream），做一次取模
        slot = int(slot) % int(ring_size)
        with self._ring_lock_sync:
            self._ring_ptr[safe_id] = slot
        print(f"[minio][warmup] safe_id={safe_id} last_slot={slot} from={obj_name}")


    async def next_ring_slot(self, safe_id: str, ring_size: int) -> Optional[int]:
        """
        返回下一个环形缓冲区槽位。

        与基类不同，使用线程锁以跨线程安全的方式维护计数器，避免 asyncio.Lock
        在不同事件循环下被重复使用导致的错误。
        """
        if ring_size <= 0:
            return None
        
        if not getattr(self,"_client", None):
            try:
                self.initialize()
            except Exception as e:
                print(f"[minio] [warmup] [warn] init failed : {e}")

        
        # ★ 新增：若首次出现该 safe_id，先用 MinIO 扫描 warmup 指针，避免重启后从 0 覆盖最新帧
        if safe_id not in self._ring_ptr:
            try:
                await self.warmup_ring_ptr_from_minio(safe_id, ring_size)
            except Exception as e:
                print(f"[minio][warmup][warn] warmup failed for {safe_id}: {e}")

        with self._ring_lock_sync:
            current = self._ring_ptr.get(safe_id, -1)
            next_index = (current + 1) % int(ring_size)
            self._ring_ptr[safe_id] = next_index
            return next_index




MINIO_MANAGER = ActionMinioManager(
    endpoint=ACTION_MINIO_ENDPOINT,
    access_key=ACTION_MINIO_ACCESS_KEY,
    secret_key=ACTION_MINIO_SECRET_KEY,
    bucket=ACTION_MINIO_BUCKET,
    secure=ACTION_MINIO_SECURE,
    lifecycle_days=ACTION_MINIO_LIFECYCLE_DAYS,
    io_executor=EXECUTOR_IO,
    upload_executor=ACTION_MINIO_UPLOAD_EXECUTOR,
)
r = redis.Redis(
    host=config_settings.redis.host,
    port=6379,
    password=None,
    db=0,
    decode_responses=False,  # 保持二进制，和你当前用法一致
)


def _read_one_from_redis(stream_key: str, block_ms: int = 1000):

    #print("get_one_frame_stream_key:"+str(stream_key))

    return r.xread({f"frames:{stream_key}": "$"}, count=1, block=block_ms) or []


async def get_one_frame(stream_key: str, timeout_sec: float = 1.0):
    
    #print("get_one_frame_stream_key:"+str(stream_key))



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
                    return msg_id, meta, jpeg, elapsed_ms
        else:
            print(f"[redis] no frame for {stream_key}, retry...")
        await asyncio.sleep(0.01)
    return None, None, None, None

def get_one_frame_sync(stream_key: str, timeout_sec: float = 1.0):
    """
    同步版本：从 Redis中读取一帧。
    
    返回: (msg_id, meta, jpeg_bytes, redis_fetch_ms)
    若超时未读到帧，返回 (None, None, None, None)
    """
    # 处理一下双前缀问题：如果已经带 frames:，就去掉再交给 _read_one_from_redis
    real_key = stream_key
    if real_key.startswith("frames:"):
        real_key = real_key[len("frames:") :]

    deadline = time.time() + timeout_sec
    start_monotonic = time.monotonic()
    block_ms = int(timeout_sec * 1000)

    while time.time() < deadline:
        # _read_one_from_redis 会自己加一层 frames:
        # → 实际读的是 key = "frames:{real_key}"
        streams = _read_one_from_redis(real_key, block_ms)
        if streams:
            elapsed_ms = (time.monotonic() - start_monotonic) * 1000.0
            print(f"[redis] got frame for {stream_key}")
            for _stream, messages in streams:
                for _id, fields in messages:
                    meta = json.loads(fields[b"meta"].decode("utf-8"))
                    jpeg = fields[b"jpeg"]
                    msg_id = _id.decode() if isinstance(_id, bytes) else str(_id)
                    return msg_id, meta, jpeg, elapsed_ms

        print(f"[redis] no frame for {stream_key}, retry...")
        time.sleep(0.01)

    # 超时
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
    max_upscale_factor: float = 2.0,
) -> Tuple[np.ndarray, float, float]:
    height, width = crop.shape[:2]
    if height == 0 or width == 0:
        return crop, 1.0, 1.0

    short_side = min(height, width)
    if short_side >= target_short_side:
        return crop, 1.0, 1.0

    raw_scale = target_short_side / short_side
    scale = min(raw_scale, max(1.0, float(max_upscale_factor)))
    if scale <= 1.0:
        return crop, 1.0, 1.0

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


def _run_coroutine_sync(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        # 仅在当前线程已有事件循环时进行兜底处理，避免对已执行过的协程二次 await。
        if "asyncio.run() cannot be called" not in str(e):
            raise

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def _ensure_minio_ready() -> bool:
    if MINIO_MANAGER.is_ready:
        return True
    MINIO_MANAGER.initialize()
    return MINIO_MANAGER.is_ready


def _encode_jpeg_bytes(image: np.ndarray, quality: int = MINIO_JPEG_QUALITY) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buf.tobytes()


def _build_frame_obj_key(
    safe_id: str, frame_idx: int, msg_id: Optional[str], annotated: bool
) -> str:
    suffix = "annotated" if annotated else "raw"
    if MINIO_RING_ENABLED and MINIO_RING_SIZE > 0:
        slot_index = _run_coroutine_sync(
            MINIO_MANAGER.next_ring_slot(safe_id, MINIO_RING_SIZE)
        )
        if slot_index is not None:
            return f"{safe_id}/ring/{slot_index:06d}_{suffix}.jpg"

    ts = datetime.utcnow().strftime("%Y%m%d/%H%M%S_%f")
    msg_part = msg_id.replace(":", "_") if msg_id else f"f{frame_idx:06d}"
    return f"{safe_id}/frames/{ts}_{msg_part}_{suffix}.jpg"


def _build_crop_obj_key(
    safe_id: str, frame_idx: int, person_id: int, msg_id: Optional[str]
) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d/%H%M%S_%f")
    msg_part = msg_id.replace(":", "_") if msg_id else f"f{frame_idx:06d}"
    return f"{safe_id}/crops/{ts}_{msg_part}_p{person_id:02d}.jpg"


def _upload_image_bytes(obj_key: str, data: Optional[bytes]) -> Optional[str]:
    if not data or not _ensure_minio_ready():
        return None
    return _run_coroutine_sync(MINIO_MANAGER.put_bytes(obj_key, data))


def _upload_frame_to_minio(
    safe_id: str, frame_idx: int, msg_id: Optional[str],
    image: np.ndarray, annotated: bool
) -> Optional[str]:
    jpeg = _encode_jpeg_bytes(image, MINIO_JPEG_QUALITY)
    obj_key = _build_frame_obj_key(safe_id, frame_idx, msg_id, annotated)
    etag = _upload_image_bytes(obj_key, jpeg)
    return obj_key if etag else None


def _build_minio_public_url(obj_key: Optional[str]) -> Optional[str]:
    """
    根据 MinIO 的 endpoint / bucket / secure 拼出一个可访问的 HTTP(S) URL。
    例如：
      endpoint = "4.1.9.10:9000"
      bucket   = "pose-action"
      obj_key  = "rtsp___192_168_130_14_8554_test2/ring/000191_annotated.jpg"

    则返回：
      "http://4.1.9.10:9000/pose-action/rtsp___192_168_130_14_8554_test2/ring/000191_annotated.jpg"
    """
    if not obj_key:
        return None

    # 1) 协议：根据 MinIO secure 判断 http / https
    scheme = "https" if ACTION_MINIO_SECURE else "http"

    # 2) endpoint 可能已经带了 http:// 或 https://，这里统一剥掉前缀
    endpoint = ACTION_MINIO_ENDPOINT
    if endpoint.startswith("http://"):
        endpoint = endpoint[len("http://") :]
    elif endpoint.startswith("https://"):
        endpoint = endpoint[len("https://") :]

    # 3) 拼接 URL：scheme://endpoint/bucket/obj_key
    return f"{scheme}://{endpoint}/{ACTION_MINIO_BUCKET}/{obj_key}"


def _upload_crop_to_minio(
    safe_id: str, frame_idx: int, person_id: int, msg_id: Optional[str], crop: np.ndarray
) -> Optional[str]:
    jpeg = _encode_jpeg_bytes(crop, MINIO_JPEG_QUALITY)
    obj_key = _build_crop_obj_key(safe_id, frame_idx, person_id, msg_id)
    etag = _upload_image_bytes(obj_key, jpeg)
    return obj_key if etag else None


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

        self._grounding_model = YOLO("/workspace/models/yolo11n.pt").to(self._device)
        #self._grounding_model = YOLO("/workspace/yolo11m.pt").to(self._device)

        self._class_names = _load_class_names(data_config) or DEFAULT_ACTION_CLASSES
        self._class_name_to_id = {
            name.lower(): idx for idx, name in self._class_names.items()
        }
        self._enable_yolo_warmup = str(os.environ.get("YOLO_WARMUP", "1")).lower() not in ("0", "false")
        self._log_detect_stats = True
        self._detect_stats_logged = False
        self._min_orig_short_side = int(os.environ.get("MIN_ORIG_SHORT_SIDE", "96"))
        self._min_blur_var = float(os.environ.get("MIN_CROP_BLUR_VAR", "25.0"))
        self._max_crop_upscale_factor = float(os.environ.get("MAX_CROP_UPSCALE_FACTOR", "2.0"))

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

        import queue  

        self._benchmark = BenchmarkRecorder()

        # 全局 crop 任务队列（生产端放 crop，消费端 vLLM Worker 消费）
        self._crop_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(
            maxsize=int(os.environ.get("CROP_QUEUE_SIZE", "2048"))
        )
        self._consumer_workers = int(os.environ.get("CROP_CONSUMER_WORKERS", "16"))
        self._consumers_started = False
        self._on_result_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        # ==== 检测侧：队列 + worker 池 ====
        self._detect_queue_size = int(os.environ.get("DETECT_QUEUE_SIZE", "20"))
        self._detect_workers = int(os.environ.get("DETECT_WORKERS", "1"))
        self._detect_batch_size = int(os.environ.get("DETECT_BATCH_SIZE", "64"))
        self._detect_batch_wait_ms = float(os.environ.get("DETECT_BATCH_WAIT_MS", "1000"))
        self._detect_workers_started = False
        self._detect_queues: Dict[str, "queue.Queue[Dict[str, Any]]"] = {}
        self._detect_queues_lock = threading.Lock()
        self._detect_rr_index = 0

        # === 新增：每路流的冷静期控制 ===
        import threading as _threading_mod
        self._cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
        # 记录每个 stream_key 的“冷静到期时间戳”（time.time() 的秒）
        self._cooldown_until: Dict[str, float] = {}
        self._cooldown_lock = _threading_mod.Lock()
        
        # === 异常类别最小置信度（再保险）：低于此阈值一律当 unknown 处理 ===
        self._abnormal_min_conf = ABNORMAL_MIN_CONFIDENCE

        if self._enable_yolo_warmup:
            self._warmup_yolo()



    def _detect_with_grounding(self, frame_pil: Image.Image) -> List[GroundingResult]:
        results = self._grounding_model(frame_pil, conf=self._grounding_threshold, classes=[0])  # 0=COCO person
        if self._log_detect_stats and not self._detect_stats_logged and results:
            try:
                r0 = results[0]
                speed = getattr(r0, "speed", {}) or {}
                print(
                    f"[detect] device={self._grounding_model.device} input_shape={r0.orig_shape} "
                    f"preprocess_ms={speed.get('preprocess', 'na')} "
                    f"inference_ms={speed.get('inference', 'na')} "
                    f"postprocess_ms={speed.get('postprocess', 'na')}"
                )
                self._detect_stats_logged = True
            except Exception as e:
                print(f"[detect] log stats failed: {e}")
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                score = float(box.conf[0].cpu())
                detections.append(GroundingResult(box_xyxy=(x1, y1, x2, y2), score=score, label="person"))
        return detections

    def _warmup_yolo(self) -> None:
        """Run a tiny forward pass so cold-start latency is paid at service startup."""
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            dummy_pil = Image.fromarray(dummy)
            with torch.no_grad():
                _ = self._grounding_model(
                    dummy_pil,
                    conf=self._grounding_threshold,
                    classes=[0],
                )
                if torch.cuda.is_available() and "cuda" in str(self._device):
                    torch.cuda.synchronize()
            print("[warmup] YOLO grounding model warmed up.")
        except Exception as e:
            print(f"[warmup] YOLO grounding model warmup failed: {e}")

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



    def _start_consumers_if_needed(self, on_result: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """
        确保 vLLM 消费端线程已启动。on_result 由 API 层传入，用于 WebSocket 推送。
        """
        # 每次最新的 on_result 覆盖旧的，方便重启
        if on_result is not None:
            self._on_result_callback = on_result

        if self._consumers_started:
            return

        self._consumers_started = True
        # 初始队列长度记录一次，方便观测启动阶段的 backlog
        try:
            self._benchmark.record("crop_queue_size", float(self._crop_queue.qsize()))
        except Exception:
            pass
        for i in range(self._consumer_workers):
            t = threading.Thread(
                target=self._consumer_loop,
                name=f"crop-consumer-{i}",
                daemon=True,
            )
            t.start()

    def _start_detect_workers_if_needed(self) -> None:
        if self._detect_workers_started:
            return
        self._detect_workers_started = True
        for i in range(self._detect_workers):
            t = threading.Thread(
                target=self._detect_worker_loop,
                name=f"detect-worker-{i}",
                daemon=True,
            )
            t.start()

    def _consumer_loop(self) -> None:
        """
        vLLM 消费端：从 _crop_queue 取出 crop，调用 _classify_action_with_qwen，然后通过 on_result 推送。
        """
        import queue as _queue_mod


        

        prefs = get_upload_preferences()   # {"upload_all_frames":..., "upload_annotated":...}
        upload_annotated = bool(prefs.get("upload_annotated", True))

        print("=============================debug====================================即将进入循环")
        while True:
            try:
                task = self._crop_queue.get(timeout=1.0)
            except _queue_mod.Empty:
                continue

            now = time.perf_counter()
            enqueue_ts = task.get("enqueue_ts", now)
            flow_start_ts = task.get("flow_start_ts", enqueue_ts)
            queue_wait_ms = (now - enqueue_ts) * 1000.0
            flow_to_consume_ms = (now - flow_start_ts) * 1000.0
            self._benchmark.record("vllm_queue_wait", queue_wait_ms)
            self._benchmark.record("pipeline_to_consume", flow_to_consume_ms)
            try:
                self._benchmark.record("crop_queue_size", float(self._crop_queue.qsize()))
            except Exception:
                pass

            crop = task["crop"]
            qwen_action = self._classify_action_with_qwen(crop)

            cls_id = qwen_action.class_id
            cls_name_norm = (qwen_action.class_name or "").strip().lower()
            conf = float(qwen_action.confidence or 0.0)

            # === 0) 统计原始输出（可选） ===
            try:
                self._benchmark.record("raw_cls_conf", conf)
            except Exception:
                pass

            # === 1) 无效标签 / 解析失败 → 直接丢弃 ===
            if cls_id is None or cls_name_norm == "":
                self._benchmark.record_drop("invalid_label_drop")
                self._crop_queue.task_done()
                continue

            if cls_id not in VALID_ID_TO_NAME or cls_name_norm not in VALID_NAME_TO_ID:
                self._benchmark.record_drop("invalid_label_drop")
                self._crop_queue.task_done()
                continue

            # 再做一层一致性验证（理论上在 _classify_action_with_qwen 已经处理过，这里防御式）
            if VALID_ID_TO_NAME.get(cls_id) != cls_name_norm:
                self._benchmark.record_drop("id_name_mismatch_drop")
                self._crop_queue.task_done()
                continue

            # === 2) unknown 一律丢弃 ===
            if cls_id == -1 or cls_name_norm in UNKNOWN_CLASSES:
                self._benchmark.record_drop("unknown_drop")
                self._crop_queue.task_done()
                continue

            

            # === 4) 不在异常集合里的 label 一律丢弃（防止未来 prompt 变化） ===
            if cls_name_norm not in ABNORMAL_CLASSES:
                self._benchmark.record_drop("non_abnormal_drop")
                self._crop_queue.task_done()
                continue

            # === 5) 置信度过滤（宁可漏检，不可错检） ===
            if conf < self._abnormal_min_conf:
                self._benchmark.record_drop("low_confidence_drop")
                self._crop_queue.task_done()
                continue

            # === 6) 冷静期逻辑（只对“已通过置信度过滤的异常”生效） ===
            stream_key = task.get("stream_key", task.get("stream_url", ""))
            now_wall = time.time()
            in_cooldown = False

            # 冷静期调整，如果是 "walk" 行为，冷静期加长
            def _get_cooldown_for_class(cls_name: str) -> float:
                cooldown = DEFAULT_COOLDOWN_SECONDS
                if cls_name == "walk":
                    cooldown *= 5  # 将 walk 行为的冷静期加长
                return cooldown

            with self._cooldown_lock:
                # 判断是否是 "walk" 行为，设置冷静期
                cooldown_period = _get_cooldown_for_class(cls_name_norm)
                cooldown_until = self._cooldown_until.get(stream_key, 0.0)
                if now_wall < cooldown_until:
                    in_cooldown = True
                else:
                    # 当前不在冷静期，则允许这次事件通过，并立即刷新冷静期
                    self._cooldown_until[stream_key] = now_wall + cooldown_period

            if in_cooldown:
                self._benchmark.record_drop("cooldown_drop")
                self._crop_queue.task_done()
                continue

            # === 走到这里：说明已经是“高置信度异常 + 不在冷静期” → 可以进入上传 & WebSocket 阶段 ===

            raw_frame = task["raw_frame"]
            bbox = task["bbox_xyxy"]

            label = f'{qwen_action.class_name or "unknown"} {qwen_action.confidence or 0:.2f}'
            if upload_annotated:
                upload_img = _draw_one_detection(raw_frame, bbox, label)
                obj_key = _upload_frame_to_minio(task["safe_id"], task["frame_index"], task["msg_id"], upload_img, annotated=True)
            else:
                obj_key = _upload_frame_to_minio(task["safe_id"], task["frame_index"], task["msg_id"], raw_frame, annotated=False)
            # ★ 新增：构造完整可访问的 URL
            image_minio_url = _build_minio_public_url(obj_key)


            flow_total_ms = (time.perf_counter() - flow_start_ts) * 1000.0
            self._benchmark.record("pipeline_total", flow_total_ms)

            # === benchmark：记录 vLLM 各阶段耗时 ===
            self._benchmark.record("vllm_encode", qwen_action.timings.get("encode_ms"))
            self._benchmark.record("vllm_build_msg", qwen_action.timings.get("build_msg_ms"))
            self._benchmark.record("vllm_http", qwen_action.timings.get("http_ms"))
            self._benchmark.record("vllm_parse", qwen_action.timings.get("parse_json_ms"))
            self._benchmark.record("vllm_total", qwen_action.timings.get("total_ms"))

            # === 构造 WebSocket payload（每个 crop 一条），附带原始帧 obj_key ===
            if self._on_result_callback is not None:
                bbox_xyxy_norm = task["bbox_xyxy_norm"]
                payload = {
                    "stream": task["stream_url"],
                    "msg_id": task["msg_id"],
                    "meta": task["meta"],
                    "frame_obj_key": obj_key,
                    "uploaded_annotated": upload_annotated,
                    # ★ 新增：对齐文档要求的完整 URL
                    "image_minio": image_minio_url,
                    "detection": {
                        "bbox_xyxy": bbox_xyxy_norm,
                        "conf": qwen_action.confidence,
                        "cls": qwen_action.class_name or "unknown",
                        "rationale": qwen_action.rationale,
                    },
                    "latency": {
                        "redis_fetch_ms": task.get("redis_fetch_ms"),
                        "decode_ms": task.get("decode_ms"),
                        "detection_ms": task.get("detection_ms"),
                        "enqueue_wait_ms": queue_wait_ms,
                        "flow_to_consume_ms": flow_to_consume_ms,
                        "pipeline_total_ms": flow_total_ms,
                        "vllm_encode_ms": qwen_action.timings.get("encode_ms"),
                        "vllm_http_ms": qwen_action.timings.get("http_ms"),
                        "vllm_parse_ms": qwen_action.timings.get("parse_json_ms"),
                        "vllm_total_ms": qwen_action.timings.get("total_ms"),
                    },
                    "frame_index": task["frame_index"],
                    "person_index": task["person_index"],
                }
                try:
                    self._on_result_callback(payload)
                except Exception as e:
                    print(f"[consumer] on_result callback failed: {e}")

            self._crop_queue.task_done()

    def _enqueue_detect_task(self, task: Dict[str, Any]) -> None:
        """入检测队列，队列满则丢最旧的任务以保持实时性。"""
        stream_key = task.get("stream_key", "")
        q = self._get_or_create_detect_queue(stream_key)
        try:
            q.put_nowait(task)
        except queue.Full:
            try:
                _ = q.get_nowait()  # 丢最旧
                self._benchmark.record_drop("detect_drop")
            except Exception:
                pass
            try:
                q.put_nowait(task)
            except Exception:
                return
        self._record_detect_queue_size()

    def _enqueue_crop_task(self, task: Dict[str, Any]) -> None:
        try:
            self._crop_queue.put_nowait(task)
        except queue.Full:
            try:
                _ = self._crop_queue.get_nowait()   # 丢最旧
                self._benchmark.record_drop("crop_drop_oldest")
                #self._crop_queue.task_done()        # 可选：如果你在别处依赖 task_done 计数
            except Exception:
                pass
            try:
                self._crop_queue.put_nowait(task)
            except Exception:
                self._benchmark.record_drop("crop_drop_new")
                return
        try:
            self._benchmark.record("crop_queue_size", float(self._crop_queue.qsize()))
        except Exception:
            pass


    def _detect_worker_loop(self) -> None:
        """检测 worker：批处理 YOLO，结果入 crop 队列。"""
        import queue as _queue_mod

        while True:
            first = self._get_next_detect_task_blocking()
            batch: List[Dict[str, Any]] = []
            if first is None:
                continue
            batch.append(first)
            batch_start = time.perf_counter()
            while len(batch) < self._detect_batch_size:
                remaining = self._detect_batch_wait_ms / 1000.0 - (time.perf_counter() - batch_start)
                if remaining <= 0:
                    break
                nxt = self._get_next_detect_task_nonblocking()
                if nxt is None:
                    time.sleep(max(0.0, min(remaining, 0.005)))
                    continue
                batch.append(nxt)

            images = []
            for t in batch:
                frame = t["frame"]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(frame_rgb))

            t_detect_start = time.perf_counter()
            results = self._grounding_model(
                images,
                conf=self._grounding_threshold,
                classes=[0],
            )
            t_detect_end = time.perf_counter()
            detection_ms = (t_detect_end - t_detect_start) * 1000.0

            for t, r in zip(batch, results):
                stream_key = t["stream_key"]
                frame_idx = t["frame_index"]
                meta = t["meta"]
                frame = t["frame"]
                img_height, img_width = frame.shape[:2]

                detections: List[GroundingResult] = []
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    score = float(box.conf[0].cpu())
                    detections.append(GroundingResult(box_xyxy=(x1, y1, x2, y2), score=score, label="person"))

                self._benchmark.record("detection", detection_ms)

                if detections:
                    cx_frame, cy_frame = img_width / 2.0, img_height / 2.0

                    def _center_dist(det: GroundingResult) -> float:
                        x1, y1, x2, y2 = det.box_xyxy
                        cx = 0.5 * (x1 + x2)
                        cy = 0.5 * (y1 + y2)
                        return (cx - cx_frame) ** 2 + (cy - cy_frame) ** 2

                    detections = sorted(detections, key=_center_dist)[:5]

                if not detections:
                    continue

                

                for person_id, detection in enumerate(detections):
                    x1, y1, x2, y2 = _expand_box(
                        detection.box_xyxy, t["box_expand_ratio"], img_width, img_height
                    )
                    crop = frame[y1:y2, x1:x2].copy()
                    # 保持原始 crop 尺寸，不做最短边上采样

                    bbox_xyxy_norm = [
                        x1 / img_width,
                        y1 / img_height,
                        x2 / img_width,
                        y2 / img_height,
                    ]

                    task = {
                        "stream_key": stream_key,
                        "stream_url": meta.get("stream", stream_key) if meta else stream_key,
                        "safe_id": t["safe_id"],
                        "frame_index": frame_idx,
                        "person_index": person_id,
                        "raw_frame": frame,
                        "bbox_xyxy": (x1, y1, x2, y2),
                        "bbox_xyxy_norm": bbox_xyxy_norm,
                        "grounding_score": float(detection.score),
                        "crop": crop,
                        "msg_id": t["msg_id"],
                        "meta": meta or {},
                        #"raw_frame_obj_key": raw_frame_obj_key,
                        "flow_start_ts": t.get("flow_start_ts", time.perf_counter()),
                        "redis_fetch_ms": t.get("redis_fetch_ms"),
                        "decode_ms": t.get("decode_ms"),
                        "detection_ms": detection_ms,
                        "enqueue_ts": time.perf_counter(),
                    }

                    self._enqueue_crop_task(task)


            for _ in batch:
                try:
                    q = self._get_or_create_detect_queue(_.get("stream_key", ""))
                    q.task_done()
                except Exception:
                    pass

    def _get_or_create_detect_queue(self, stream_key: str) -> "queue.Queue[Dict[str, Any]]":
        with self._detect_queues_lock:
            q = self._detect_queues.get(stream_key)
            if q is None:
                q = queue.Queue(maxsize=self._detect_queue_size)
                self._detect_queues[stream_key] = q
            return q

    def _record_detect_queue_size(self) -> None:
        try:
            with self._detect_queues_lock:
                total = sum(q.qsize() for q in self._detect_queues.values())
            self._benchmark.record("detect_queue_size", float(total))
        except Exception:
            pass

    def _get_next_detect_task_blocking(self) -> Optional[Dict[str, Any]]:
        deadline = time.perf_counter() + 1.0
        while True:
            task = self._get_next_detect_task_nonblocking()
            if task is not None:
                return task
            if time.perf_counter() > deadline:
                return None
            time.sleep(0.005)

    def _get_next_detect_task_nonblocking(self) -> Optional[Dict[str, Any]]:
        with self._detect_queues_lock:
            stream_keys = list(self._detect_queues.keys())
            if not stream_keys:
                return None
            start_idx = self._detect_rr_index % len(stream_keys)

        total = len(stream_keys)
        for i in range(total):
            sk = stream_keys[(start_idx + i) % total]
            q = self._detect_queues.get(sk)
            if q is None:
                continue
            try:
                task = q.get_nowait()
                self._record_detect_queue_size()
                self._detect_rr_index = start_idx + i + 1
                return task
            except queue.Empty:
                continue
        return None

    def benchmark_snapshot(self) -> Dict[str, Dict[str, float]]:
        """给 API 层 / 脚本使用，返回当前运行期间各阶段的耗时统计。"""
        return self._benchmark.snapshot()


    def _classify_action_with_qwen(self, crop: np.ndarray) -> QwenAction:
        # === (0) 总计时开始 ===
        t_start = time.perf_counter()

        # === (0.5) 质量门控：远处小目标 / 严重模糊直接 unknown，避免上采样“脑补” ===
        h, w = crop.shape[:2]
        short_side = min(h, w) if h > 0 and w > 0 else 0
        blur_var = 0.0
        if h > 0 and w > 0:
            try:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            except Exception:
                blur_var = 0.0

        if short_side < self._min_orig_short_side or blur_var < self._min_blur_var:
            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0
            rationale = "target too small or blurry"
            return QwenAction(
                class_id=-1,
                class_name="unknown",
                confidence=0.2,
                rationale=rationale,
                raw_response=json.dumps({
                    "class_id": -1,
                    "class_name": "unknown",
                    "confidence": 0.2,
                    "rationale": rationale,
                }, ensure_ascii=False),
                timings={
                    "encode_ms": 0.0,
                    "build_msg_ms": 0.0,
                    "http_ms": 0.0,
                    "parse_json_ms": 0.0,
                    "total_ms": total_ms,
                    "quality_gate_hit": 1.0,
                },
                meta={
                    "jpeg_bytes": 0,
                    "crop_wh": (w, h),
                    "blur_var": blur_var,
                    "short_side": short_side,
                },
            )

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
            
        # === (5.5) 代码级安全兜底：强制校验 id/name 一致性 & 合法性 ===
        cls_id = class_id
        cls_name = (class_name or "").strip().lower()

        # 1) 先把名称归一到合法集合
        if cls_name in VALID_NAME_TO_ID:
            # 如果 name 合法但 id 缺失/不一致，用 name 反推 id
            mapped_id = VALID_NAME_TO_ID[cls_name]
            if cls_id is None or cls_id != mapped_id:
                cls_id = mapped_id
        elif cls_id in VALID_ID_TO_NAME:
            # 如果 id 合法但 name 非法，用 id 反推 name
            cls_name = VALID_ID_TO_NAME[cls_id]
        else:
            # 两个都不在合法集合里 → 强制 unknown
            cls_id = -1
            cls_name = "unknown"

        # 2) 再做一轮最终合法性检查
        if cls_id not in VALID_ID_TO_NAME or cls_name not in VALID_NAME_TO_ID:
            cls_id = -1
            cls_name = "unknown"

        # 3) 置信度兜底
        if confidence is None:
            confidence = 0.2
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.2

        # 约束到 [0,1] 之间
        confidence = max(0.0, min(1.0, confidence))

        # unknown 的时候强制把置信度压到 0.2，表示“不确定”
        if cls_id == -1 or cls_name in UNKNOWN_CLASSES:
            confidence = 0.2

        class_id = cls_id
        class_name = cls_name

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

        self._benchmark.record("vllm_encode", encode_ms)
        self._benchmark.record("vllm_build_msg", build_msg_ms)
        self._benchmark.record("vllm_http", http_ms)
        self._benchmark.record("vllm_parse", parse_json_ms)
        self._benchmark.record("vllm_total", total_ms)

        return QwenAction(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            rationale=rationale,
            raw_response=response_text,
            timings=timings,
            meta=meta,
        )


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

        # 启动 vLLM 消费端线程，并把 on_result 传下去
        self._start_consumers_if_needed(on_result)
        # 启动检测 worker 池
        self._start_detect_workers_if_needed()


        """
        从 Redis 中的帧流（frames:{stream_name}）读取视频帧进行处理。
        """
        safe_id = safe_filename(stream_key)
        save_root = Path(output_dir) if output_dir and save_video else None
        if save_root and save_video:
            save_root.mkdir(parents=True, exist_ok=True)

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

                # === 3. 从 Redis 同步读一帧（使用新的 get_one_frame_sync） ===
                fetch_start = time.perf_counter()
                msg_id, meta, jpeg, redis_fetch_ms = get_one_frame_sync(
                    stream_key, timeout_sec=timeout_sec
                )
                fetch_end = time.perf_counter()
                # 如果 get_one_frame_sync 没给出耗时，就这里兜底算一次（只在成功拿到帧时有意义）
                if jpeg is not None:
                    if redis_fetch_ms is None:
                        redis_fetch_ms = (fetch_end - fetch_start) * 1000.0
                    # 只在有帧时记录 redis_fetch
                    if redis_fetch_ms is not None:
                        self._benchmark.record("redis_fetch", redis_fetch_ms)
                else:
                    # 超时未拿到帧，直接下一轮
                    print(f"[stream {stream_key}] timeout {timeout_sec}s, no frame.")
                    continue

                frame_idx += 1
                if progress is not None:
                    progress.update(1)

                if frame_idx % frame_interval != 0:
                    continue

                decode_start = time.perf_counter()

                np_buf = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                decode_end = time.perf_counter()
                decode_ms = (decode_end - decode_start) * 1000.0
                self._benchmark.record("decode", decode_ms)
                if frame is None:
                    print(f"[stream {stream_key}] decode failed, skip this frame.")
                    
                        
                    continue




                detect_task = {
                    "stream_key": stream_key,
                    "safe_id": safe_id,
                    "frame_index": frame_idx,
                    "msg_id": msg_id,
                    "meta": meta or {},
                    "frame": frame,
                    "box_expand_ratio": box_expand_ratio,
                    "crop_min_short_side": crop_min_short_side,
                    "flow_start_ts": fetch_start,
                    "redis_fetch_ms": redis_fetch_ms,
                    "decode_ms": decode_ms,
                }
                self._enqueue_detect_task(detect_task)

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
    "set_upload_all_frames",
    "set_upload_annotated",
    "get_upload_preferences",
    "ACTIVITY_TRACKER",
]
