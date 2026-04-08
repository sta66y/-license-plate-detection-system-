FROM ultralytics/ultralytics:latest

WORKDIR /app

RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84

COPY src/ ./src/
COPY pyproject.toml ./

RUN mkdir -p /app/data /app/videos /app/output /app/models

ENV PYTHONPATH=/app
ENV LOG_DIR=/app/data

CMD ["python", "src/main.py", "--help"]