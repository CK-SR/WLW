# FROM hqwlw_zzs:v0
FROM hqwlw-slim:v2

# 工作目录
WORKDIR /workspace

# 清理旧代码目录，再拷贝新的 wlw 工程
RUN rm -rf /workspace
COPY hqwlw /workspace

## Python 输出不缓冲，便于日志采集
#ENV PYTHONDONTWRITEBYTECODE=1 \
#    PYTHONUNBUFFERED=1
#
## 可选：仅用于文档化端口（真正映射放到 docker-compose）
#EXPOSE 8000
