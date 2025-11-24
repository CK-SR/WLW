from __future__ import annotations

import asyncio
from typing import Any, Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
clients: Set[WebSocket] = set()
clients_lock = asyncio.Lock()


async def broadcast(message: Dict[str, Any]) -> None:
    dead = []
    async with clients_lock:
        for ws in list(clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.remove(ws)


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    async with clients_lock:
        clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with clients_lock:
            clients.discard(ws)
