FROM python:3.11-slim

WORKDIR /code

# CPU-only torch/torchvision wheels, installed separately from
# requirements.txt to avoid pulling in ~2GB of unused CUDA libraries.
# Cloud Run's free tier is CPU-only, so a GPU build buys nothing here.
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Cloud Run injects $PORT at runtime; default to 8080 for local testing.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
