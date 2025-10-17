import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Callable
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute, APIWebSocketRoute
from pydantic import BaseModel
from contextlib import AsyncExitStack, asynccontextmanager
import sys, logging
from pathlib import Path
# =============== 导入所有业务 router 和 lifespan ===============
from areaIntrusion_fastapi.src.area_interface import router as area_router
from areaIntrusion_fastapi.src.area_interface import lifespan as area_lifespan
from fire_fastapi.fire_interface_redis import router as fire_redis_router
from camera_check_fastapi.src.main import router as camera_check_router
from face_fastapi.face_redis import router as face_redis_router
from face_fastapi.face_image_ebd_updata import router as face_image_ebd_updata_router
from dashboard_fastapi.dashboard_redis import router as dashboard_redis_router
from pose_fastapi_new.pose_interface import router as pose_pipeline_router
from configs.settings import settings as config_settings

LOG_DIR = getattr(getattr(config_settings, "logging", None), "dir", "/var/log/app")
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
handlers = [
    logging.StreamHandler(sys.stdout),                         # 控制台
    logging.FileHandler(Path(LOG_DIR) / "app.log", encoding="utf-8"),  # 文件
]
logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)

# =============== 统一注册表（router + lifespan） ===============
ROUTER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "area": {"router": area_router, "lifespan": area_lifespan},
    "pose_pipeline": {"router": pose_pipeline_router},
    "fire_redis": {"router": fire_redis_router},
    "camera": {"router": camera_check_router},
    "face": {"router": face_redis_router},
    "face_ebd": {"router": face_image_ebd_updata_router},
    "dashboard": {"router": dashboard_redis_router},
}


class StreamListRequest(BaseModel):
    streams: List[Dict[str, Any]]


def _parse_enabled_from_env() -> list[str]:
    """从环境变量 ENABLED_ROUTERS 解析启用列表，逗号分隔"""
    val = os.getenv("ENABLED_ROUTERS", "")
    if not val:
        return []
    return [x.strip().lower() for x in val.split(",") if x.strip()]


def _collect_lifespans(enabled: list[str]):
    """收集启用的 lifespan"""
    lifespans = []
    for name in enabled:
        entry = ROUTER_REGISTRY.get(name)
        if entry and "lifespan" in entry:
            lifespans.append(entry["lifespan"])
    return lifespans


def _merge_lifespans(lifespans: list[Callable]):
    """把多个 lifespan 合并成一个总 lifespan"""
    if not lifespans:
        return None

    @asynccontextmanager
    async def combined_lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            for lf in lifespans:
                await stack.enter_async_context(lf(app))
            yield

    return combined_lifespan


def create_app() -> FastAPI:
    # 解析启用列表，默认启用全部
    enabled = _parse_enabled_from_env()
    if not enabled:
        enabled = list(ROUTER_REGISTRY.keys())
        print(f"[BOOT] No ENABLED_ROUTERS set, enabling ALL: {enabled}")
    else:
        print(f"[BOOT] ENABLED_ROUTERS: {enabled}")

    lifespans = _collect_lifespans(enabled)
    combined_lifespan = _merge_lifespans(lifespans)

    app = FastAPI(
        title="Unified API Gateway",
        version="1.3.0",
        lifespan=combined_lifespan
    )

    included = []
    for name in enabled:
        entry = ROUTER_REGISTRY.get(name)
        if entry and "router" in entry:
            app.include_router(entry["router"])
            included.append(name)
        else:
            print(f"[WARN] Unknown router '{name}', skip")

    # ✅ 在这里打印最终启用的 router 清单
    print(f"[BOOT] Final enabled routers: {included}")
    # =============== 调试用：列出当前注册路由 ===============
    @app.get("/__routes")
    def __routes():
        http_routes = []
        ws_routes = []
        for r in app.router.routes:
            if isinstance(r, APIRoute):
                http_routes.append({"path": r.path, "methods": sorted(list(r.methods))})
            elif isinstance(r, APIWebSocketRoute):
                ws_routes.append({"path": r.path})
        return {"http": http_routes, "ws": ws_routes}

    @app.get("/__echo")
    async def __echo(req: Request):
        scope = req.scope
        return {
            "root_path": scope.get("root_path"),
            "path": scope.get("path"),
            "url_path_for_ws": "/fire/ws"
        }

    return app


app = create_app()

# =============== 本地启动（可选） ===============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("multi_interface:app", host="0.0.0.0", port=5011, reload=True)
