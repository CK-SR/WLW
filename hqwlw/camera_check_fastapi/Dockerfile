FROM hqwlw-slim:v2

WORKDIR /workspace

# 拷贝代码到容器
COPY . /workspace

# 安装依赖
#RUN pip install --no-cache-dir fastapi uvicorn redis opencv-python-headless numpy

# 默认启动
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5020", "--log-level", "info"]
