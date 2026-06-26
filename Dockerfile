# G-TURF web UI container.
#
# Build:  docker build -t gturf-app .
# Run:    docker run -p 7860:7860 gturf-app
# Then open http://localhost:7860
#
# Deploys as-is to Hugging Face Spaces (Docker SDK) — it exposes port 7860.
FROM python:3.11-slim

# System libs needed by matplotlib / openpyxl
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml requirements.txt README.md ./
COPY gturf ./gturf
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir "gradio>=4.0"

# App code
COPY app.py gturf_ui_helpers.py ./

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
