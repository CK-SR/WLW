import os

# 默认检测参数
DEFAULT_CURRENT_THRESHOLD = 10  # 黑屏阈值
DEFAULT_EDGE_PARAMS = {"low": 50, "high": 150, "min_ratio": 0.02}
DEFAULT_TAMPER_PARAMS = {
    "entropy_threshold": 3.0,  # 熵阈值
    "diff_threshold": 50       # 帧间差异阈值
}

# Redis 连接配置（从环境变量中获取）
REDIS_HOST = os.getenv("REDIS_HOST", "192.168.130.162")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
