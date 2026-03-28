FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run the Phase 8 Momentum trading engine
CMD ["python", "-m", "live.runner_0dte", "--strategy", "momentum", "--mode", "paper"]
