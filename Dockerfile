FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV FLAGS_use_mkldnn=false
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir \
    numpy==1.26.4 \
    paddlepaddle==2.6.2 \
    paddleocr==2.7.3 \
    opencv-python-headless==4.8.1.78 \
    fastapi \
    uvicorn \
    python-multipart \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD ["python", "ocr_server.py"]