from pathlib import Path
import os, yaml
from pydantic import BaseModel
from typing import Optional

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

class LoggingSettings(BaseSettings):
    dir: str = "/workspace/logs"
    level: str = "info"


class KafkaSettings(BaseSettings):
    bootstrap_servers: str = "192.168.130.111:9092"

class Settings(BaseModel):
    env: str
    redis: RedisConfig
    models: ModelsConfig
    server: ServerConfig
    server: ServerSettings = ServerSettings()
    logging: LoggingSettings = LoggingSettings()
    kafka: KafkaSettings =  KafkaSettings()

def load_settings() -> Settings:
    env = os.getenv("APP_ENV", "local")
    config_path = PROJECT_ROOT / "configs" / f"settings_{env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Settings(**data)

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