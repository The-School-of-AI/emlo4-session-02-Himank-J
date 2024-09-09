FROM python:3.8-slim-buster as builder
 
WORKDIR /workspace
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
 
RUN pip install --no-cache-dir --user torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
 
FROM python:3.8-slim-buster
 
WORKDIR /workspace
 
COPY --from=builder /root/.local /root/.local
 
ENV PATH=/root/.local/bin:$PATH
 
COPY train.py .
 
CMD ["python", "train.py"]