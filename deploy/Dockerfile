FROM python:3.12-slim

WORKDIR /app

# System deps for pandas/numpy compilation and ibapi
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt ibapi orjson

# Copy application code
COPY . .

# Remove deploy dir from image (not needed at runtime)
RUN rm -rf deploy/

# Create runtime directories
RUN mkdir -p /app/tokens /app/logs /app/data

ENV QUESTRADE_TOKEN_FILE=/app/tokens/.questrade_token.json
ENV PYTHONUNBUFFERED=1
# Suppress webbrowser.open() in headless container
ENV DISPLAY=

EXPOSE 8050

# Default: run paper trading with dashboard
# TRADING_MODE env var (from docker-compose) overrides this
CMD ["sh", "-c", "python start.py --mode ${TRADING_MODE:-paper}"]
