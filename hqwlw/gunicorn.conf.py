# gunicorn.conf.py
from pathlib import Path
from configs.settings import settings as config_settings  # 直接用你现有的配置对象

# 直接从 config_settings 取值，如果不存在就给默认值
def get(attr_path, default):
    cur = config_settings
    for part in attr_path.split("."):
        if hasattr(cur, part):
            cur = getattr(cur, part)
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur

bind     = f"{config_settings.server.host}:{config_settings.server.port}"
workers  = config_settings.server.workers
threads  = config_settings.server.threads
timeout  = config_settings.server.timeout
LOG_DIR  = config_settings.logging.dir
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
loglevel = config_settings.logging.level


# 日志配置：写到文件
preload_app  = True
worker_class = "uvicorn.workers.UvicornWorker"

# 关键：写文件 + 控制台同时启用
accesslog = str(Path(LOG_DIR) / "gunicorn.access.log")
errorlog  = str(Path(LOG_DIR) / "gunicorn.error.log")  # 只文件

capture_output = False  # 保留 stdout，控制台可看到