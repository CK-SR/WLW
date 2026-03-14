

# 一、MinioManager 能做什么

* ✅ **initialize()**
  连接 MinIO；若 bucket 不存在则创建；应用**生命周期策略**；设置**公共只读 bucket policy**（匿名可 `GetObject`）。

* ✅ **put_bytes(obj_key, data, content_type="image/jpeg") → etag | None**
  异步上传字节对象（如 JPEG），放到上传线程池里执行。

* ✅ **next_ring_slot(safe_id, ring_size) → int | None**
  获取下一次写入的**槽位索引**（`0..ring_size-1`）。内部带指针互斥锁与 Redis 持久化（`hset minio:ring_ptr`）。若 `ring_size<=0` 返回 `None`。

* ✅ **warmup_ring_ptr(safe_id, ring_size)**
  服务重启后从 **Redis** 或 **MinIO 现有对象**恢复 ring 指针（扫描 `{safe_id}/ring/`，找 last_modified 最新对象并解析槽位）。

* ✅ **ring_object_count(safe_id, at_least=None) → int**
  统计 `{safe_id}/ring/` 下对象数；支持“**早停**”（当达到 `at_least` 立即返回，减少列举开销）。

* ✅ **plan_shrink_no_migrate(safe_id, old_M, new_N)**
  **无迁移缩环**：立即把写环切到 `N`；**阶段1**立刻删除“最近 N 之外”的旧段；“最近 N 中跨零的高段(≥N)”**暂存**——在后续**累计写满 N 帧**后，由 `on_post_write()` 触发**阶段2**删除这段；最后把指针落位到 `P % N`。`new_N==0` 时清空该路并停写。

* ✅ **on_post_write(safe_id)**
  每次写入成功后调用一次；当“累计写满 N 帧”时自动触发**阶段2**删除，并清理缩环计划。

---

# 二、关键名词/约定

* **safe_id**：由 `safe_filename(stream_name)` 生成，仅包含 `[a-zA-Z0-9_-]`。
* **ring 键空间**：`{safe_id}/ring/000000[_intrussion].jpg`…（是否带 `_intrussion` 由全局 `MINIO_USE_INTRUSSION_SUFFIX` 决定）。
* **ring 指针**：上一帧写入槽位，内存保存在 `MinioManager._ring_ptr[safe_id]`，持久化到 Redis 的 `minio:ring_ptr`。
* **ring_size**：每路最多保留帧数（由**调用方**决定并传入）。

---

# 三、对接步骤（给其他服务）

## 1）构造与初始化

```python
from concurrent.futures import ThreadPoolExecutor
# 建议从你项目中直接 import MinioManager（路径按你的工程调整）
from camera_check_fastapi.src.main import MinioManager

EXECUTOR_IO = ThreadPoolExecutor(max_workers=64)
UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=32)

manager = MinioManager(
    endpoint="minio:9000",        # 例：主机:端口
    access_key="minioadmin",
    secret_key="minioadmin",
    bucket="frames",
    secure=False,                 # https 则 True
    lifecycle_days=3,             # >0 才会下发生命周期策略
    io_executor=EXECUTOR_IO,
    upload_executor=UPLOAD_EXECUTOR,
    monitor_callback=None,        # 可接你的监控回调
)
manager.initialize()
assert manager.is_ready, "MinIO 未就绪"
```

> `initialize()` 会：**建桶 → 生命周期策略 → 公共只读策略**。

## 2）准备 `safe_id` 与 `ring_size`

```python
import re

def safe_filename(stream_name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', "_", stream_name)

def build_ring_obj_key(safe_id: str, slot_index: int, use_intrussion_suffix: bool=False) -> str:
    tail = "_intrussion" if use_intrussion_suffix else ""
    return f"{safe_id}/ring/{slot_index:06d}{tail}.jpg"

stream_name = "camera/foyer#1"
safe_id = safe_filename(stream_name)   # e.g. "camera_foyer_1"
ring_size = 300                        # 该路最多保留 300 帧（你自己定）
```

## 3）服务重启后的指针恢复（强烈建议首次写入前做一次）

```python
import asyncio
asyncio.run(manager.warmup_ring_ptr(safe_id, ring_size))
```

## 4）常规按环写入（每帧）

```python
import asyncio

async def upload_one_jpeg(jpeg_bytes: bytes, is_intrusion: bool=False):
    slot = await manager.next_ring_slot(safe_id, ring_size)
    if slot is None:
        return None  # ring_size<=0 或未初始化
    key = build_ring_obj_key(safe_id, slot, use_intrussion_suffix=is_intrusion)
    etag = await manager.put_bytes(key, jpeg_bytes, content_type="image/jpeg")
    if etag:
        # 如果当前存在缩环计划，写满 N 帧后会由 on_post_write 触发阶段2删除
        await manager.on_post_write(safe_id)
        return key
    return None
```

> `next_ring_slot()` 内部带互斥锁与 Redis 持久化；与缩环的指针落位不会冲突。

## 5）缩环（无迁移）（把 M → N）

> **不复制数据**；阶段1立即删除“最近 N 之外”的旧段；**阶段2延迟删除**“最近 N 内跨零的高段(≥N)”——由后续写满 N 帧后触发。

```python
async def shrink_no_migrate(old_M: int, new_N: int):
    await manager.plan_shrink_no_migrate(safe_id, old_M=old_M, new_N=new_N)
    # 之后你照常调用 upload_one_jpeg(...) 写帧；
    # 记得 on_post_write 在 put_bytes 成功后被调用，以便写满 N 时自动阶段2清理
```

> **清空该路**：`new_N == 0` 即可删除 `{safe_id}/ring/` 所有对象并将指针置 -1。

## 6）统计数量（便捷工具）

```python
# 早停统计（比如只关心是否已 ≥100）
n_fast = asyncio.run(manager.ring_object_count(safe_id, at_least=100))
# 精确统计
n_full = asyncio.run(manager.ring_object_count(safe_id))
```

---

# 四、最简**可运行示例**（独立脚本）

> 目标：初始化 → 恢复指针 → 连续写 10 帧 → 缩环 300→50（无迁移）→ 再写 5 帧（以触发阶段2计数）→ 打印对象数。

```python
#!/usr/bin/env python3
import os, asyncio, re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import numpy as np
from PIL import Image

from camera_check_fastapi.src.main import MinioManager

def safe_filename(x: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', "_", x)

def build_ring_obj_key(safe_id: str, slot_index: int, use_intrussion_suffix: bool=False) -> str:
    tail = "_intrussion" if use_intrussion_suffix else ""
    return f"{safe_id}/ring/{slot_index:06d}{tail}.jpg"

def make_dummy_jpeg(w=320, h=180) -> bytes:
    arr = (np.random.rand(h, w, 3) * 255).astype("uint8")
    img = Image.fromarray(arr)  # RGB
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

async def main():
    EXECUTOR_IO = ThreadPoolExecutor(max_workers=32)
    UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=16)

    manager = MinioManager(
        endpoint=os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        bucket=os.getenv("MINIO_BUCKET", "frames"),
        secure=bool(int(os.getenv("MINIO_SECURE", "0"))),
        lifecycle_days=int(os.getenv("MINIO_LIFECYCLE_DAYS", "3")),
        io_executor=EXECUTOR_IO,
        upload_executor=UPLOAD_EXECUTOR,
        monitor_callback=None,
    )
    manager.initialize()
    assert manager.is_ready

    stream_name = "camera/foyer#1"
    safe_id = safe_filename(stream_name)
    ring_size = 300

    await manager.warmup_ring_ptr(safe_id, ring_size)

    # 写 10 帧
    for i in range(10):
        slot = await manager.next_ring_slot(safe_id, ring_size)
        key = build_ring_obj_key(safe_id, slot)
        jpeg = make_dummy_jpeg()
        etag = await manager.put_bytes(key, jpeg, "image/jpeg")
        if etag:
            await manager.on_post_write(safe_id)
        print("PUT", key)

    # 缩环 300 -> 50（无迁移）
    await manager.plan_shrink_no_migrate(safe_id, old_M=300, new_N=50)
    print("shrink no-migrate done")

    # 写 5 帧（继续环写；若存在延迟段，累计写满 N 会触发阶段2删除）
    for i in range(5):
        slot = await manager.next_ring_slot(safe_id, 50)
        key = build_ring_obj_key(safe_id, slot)
        etag = await manager.put_bytes(key, make_dummy_jpeg(), "image/jpeg")
        if etag:
            await manager.on_post_write(safe_id)
        print("PUT", key)

    # 数量（仅演示）
    count = await manager.ring_object_count(safe_id)
    print("current objects:", count)

if __name__ == "__main__":
    asyncio.run(main())
```

---

# 五、集成要点 & 常见坑

* **必须在 async 上下文里调用**：`put_bytes`、`next_ring_slot`、`plan_shrink_no_migrate`、`ring_object_count` 都是 `await`。
* **指针一致性**：`next_ring_slot` 与缩环的“指针落位”用 `_ring_lock` 协调，调用方无需加锁。
* **缩环（无迁移）语义**：

  * 阶段1删除是**立即执行**（当前实现会阻塞调用方 `await`）；
  * 阶段2依赖**后续写入**（每次成功写后记得 `await on_post_write(safe_id)`）；
  * 想要**秒回**可把阶段1删除改为**后台任务**或**改批量删除**（这部分属于服务内部优化，调用侧不用改）。
* **ring_size 的来源**：这个值由你的服务自己管理（比如从数据库 / 配置中心下发）。改小 ring_size 时请调用 `plan_shrink_no_migrate`。
* **对象命名**：如需给异常帧加后缀，按需传 `use_intrussion_suffix=True`（或沿用全局 `MINIO_USE_INTRUSSION_SUFFIX`）。
* **计数优化**：只想知道“是否已 ≥K”，用 `ring_object_count(..., at_least=K)` 能明显减少列举时间。

