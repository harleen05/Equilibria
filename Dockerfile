FROM python:3.11-slim

WORKDIR /app

# Install deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source
COPY . .

# HF Spaces runs as non-root — set permissions
RUN chmod -R 755 /app

EXPOSE 8000

# HEALTHCHECK — HF Space waits for this before marking container ready
# Without this, the automated /reset ping can arrive before uvicorn is listening
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "environment.server.main:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "/app"]