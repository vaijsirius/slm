# CPU-only image that generates a story from release/ when the container starts.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# CPU-only PyTorch wheel (much smaller than the default CUDA build), then the other tiny dependencies
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install regex einops numpy
COPY minilm/ minilm/
COPY inference.py .
COPY release/ release/
ENTRYPOINT ["python", "inference.py", "--model-dir", "release", "--device", "cpu"]
CMD ["--prompt", "Once upon a time", "--max-new-tokens", "256", "--temperature", "0.8", "--top-p", "0.9", "--seed", "42"]
