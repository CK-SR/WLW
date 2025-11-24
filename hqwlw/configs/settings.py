from pathlib import Path
import os, yaml
from pydantic import BaseModel, Field
from typing import Optional, Literal

from pydantic_settings import BaseSettings

# 自动推导项目根目录：优先 APP_ROOT > cwd > 文件所在目录的父目录
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.getenv("APP_ROOT", Path.cwd()))

if not (PROJECT_ROOT / "configs").exists():
    PROJECT_ROOT = DEFAULT_ROOT

class ModelsConfig(BaseModel):
    insightface_root: str
    insightface_name: str
    yolo_path: str
    face_data: str
    # 可选：姿态/行为类别名称映射（按索引顺序）
    pose_class_names: list[str] = Field(default_factory=list)
    # 可选：环境异常检测类别名称映射（按索引顺序），如 ["fire","smoke","water"]
    fire_class_names: list[str] = Field(default_factory=list)

class RedisConfig(BaseModel):
    host: str
    port: int
    db: int
    password: Optional[str] = None
    decode_responses: Optional[bool] = False

class ServerConfig(BaseModel):
    host: str
    port: int


class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 5011
    workers: int = 1
    threads: int = 5
    timeout: int = 0

class MinIOConfig(BaseSettings):
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    
class LoggingSettings(BaseSettings):
    dir: str = "/workspace/logs"
    level: str = "info"


class KafkaSettings(BaseSettings):
    bootstrap_servers: str = "192.168.130.111:9092"

class I18NSettings(BaseModel):
    default_language: str = "en"
    fallback_language: str = "en"
    supported_languages: list[str] = Field(default_factory=lambda: ["en"])

class MinioSettings(BaseModel):
    endpoint: str = "192.168.130.162:9100"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin123"
    bucket: str = "camera-checkout"
    secure: bool = False
    ring_enabled: bool = False
    max_frames_per_stream: int = 1000
    trim_interval: int = 400
    use_intrussion_suffix: bool = True
    save_mode: Literal["all", "abnormal", "sample", "none"] = "all"
    sample_fps: float = 1.0
    jpeg_quality: int = 85


class StreamSettings(BaseModel):
    ws_send_mode: Literal["all", "abnormal"] = "all"


class MonitoringSettings(BaseModel):
    enabled: bool = True
    history_size: int = 120
    frame_history: int = 60

class FramehubSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080


class Settings(BaseModel):
    env: str
    redis: RedisConfig
    models: ModelsConfig
    #server: ServerConfig
    server: ServerSettings = ServerSettings()
    logging: LoggingSettings = LoggingSettings()
    kafka: KafkaSettings = KafkaSettings()
    i18n: I18NSettings = I18NSettings()
    minio: MinioSettings = MinioSettings()
    stream: StreamSettings = StreamSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    framehub: FramehubSettings = FramehubSettings()


def _apply_env_overrides(settings: Settings) -> Settings:
    """Allow legacy environment variables to override YAML configuration."""

    env_map = {
        "minio.endpoint": ("MINIO_ENDPOINT", str),
        "minio.access_key": ("MINIO_ACCESS_KEY", str),
        "minio.secret_key": ("MINIO_SECRET_KEY", str),
        "minio.bucket": ("MINIO_BUCKET", str),
        "minio.secure": ("MINIO_SECURE", lambda v: str(v).lower() in {"1", "true", "yes"}),
        "minio.ring_enabled": ("MINIO_RING_ENABLED", lambda v: str(v).lower() in {"1", "true", "yes"}),
        "minio.max_frames_per_stream": ("MINIO_MAX_FRAMES_PER_STREAM", int),
        "minio.trim_interval": ("MINIO_TRIM_INTERVAL", int),
        "minio.use_intrussion_suffix": ("MINIO_USE_INTRUSSION_SUFFIX", lambda v: str(v).lower() in {"1", "true", "yes"}),
        "minio.save_mode": ("MINIO_SAVE_MODE", str),
        "minio.sample_fps": ("MINIO_SAMPLE_FPS", float),
        "minio.jpeg_quality": ("MINIO_JPEG_QUALITY", int),
        "stream.ws_send_mode": ("WS_SEND_MODE", str),
    }

    for dotted_key, (env_var, caster) in env_map.items():
        raw = os.getenv(env_var)
        if raw is None:
            continue
        section, attr = dotted_key.split(".", 1)
        target = getattr(settings, section, None)
        if target is None:
            continue
        try:
            value = caster(raw)
        except Exception:
            continue
        setattr(target, attr, value)

    return settings

def load_settings() -> Settings:
    env = os.getenv("APP_ENV", "local")
    config_path = PROJECT_ROOT / "configs" / f"settings_{env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    settings = Settings(**data)
    return _apply_env_overrides(settings)

settings = load_settings()

if __name__ == "__main__":
    import json

    forced_root = DEFAULT_ROOT
    print(f"[DEBUG] Forcing workdir -> {forced_root}")
    os.chdir(forced_root)
    settings = load_settings()

    print("=== Settings Debug ===")
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"APP_ENV      = {settings.env}")
    print("Parsed Settings:")
    print(settings.model_dump_json(indent=2))

    print(settings.kafka.bootstrap_servers)
