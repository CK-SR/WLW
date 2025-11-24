from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, BaseSettings, Field


class RedisSettings(BaseModel):
    host: str = "192.168.130.14"
    port: int = 6379
    stream_prefix: str = "frames:"


class VLLMSettings(BaseModel):
    base_url: str = "http://192.168.130.14:8010/v1"
    model: str = "/model/Qwen3-VL-8B-Instruct"
    max_workers: int = 8


class YOLOSettings(BaseModel):
    model_path: str = "/workspace/models/yolo11m.pt"
    conf: float = 0.7
    device: Optional[str] = None


class PromptSettings(BaseModel):
    action_classes_path: Optional[str] = None
    system_path: Optional[str] = None
    user_path: Optional[str] = None


class ThresholdSettings(BaseModel):
    crop_min_short_side: int = 256
    box_expand_ratio: float = 0.4
    top_k: int = 1


class PipelineSettings(BaseModel):
    max_frames: Optional[int] = None
    frame_interval: int = 1
    timeout_sec: float = 1.0


class AppSettings(BaseSettings):
    # yaml path optional
    config_path: Optional[str] = Field(default=None, env="CONFIG_PATH")

    # sub settings
    redis: RedisSettings = RedisSettings()
    vllm: VLLMSettings = VLLMSettings()
    yolo: YOLOSettings = YOLOSettings()
    prompt: PromptSettings = PromptSettings(
        action_classes_path="/workspace/action_detection_vllm/prompt/data.yaml",
        system_path="/workspace/action_detection_vllm/prompt/system.yaml",
        user_path="/workspace/action_detection_vllm/prompt/user.yaml",
    )
    thresholds: ThresholdSettings = ThresholdSettings()
    pipeline: PipelineSettings = PipelineSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def load_yaml(cls, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
        except Exception:
            return {}


@lru_cache()
def get_settings() -> AppSettings:
    # prefer env CONFIG_PATH then config/config.yaml
    env_path = os.environ.get("CONFIG_PATH")
    if env_path:
        data = AppSettings.load_yaml(Path(env_path))
        return AppSettings(**data)

    default_path = Path(__file__).resolve().parent / "config.yaml"
    data = AppSettings.load_yaml(default_path)
    return AppSettings(**data)


settings = get_settings()
