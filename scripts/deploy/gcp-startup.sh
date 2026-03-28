#!/bin/bash
# Startup script for Google Compute Engine VM
# Phase 8 Momentum Strategy deployment

set -e

# Install Python and git
apt-get update
apt-get install -y python3.11 python3.11-venv python3-pip git

# Clone or update repo
cd /opt
if [ -d "trading-engine" ]; then
    cd trading-engine
    git pull
else
    git clone https://github.com/UrbanlyticsSolutions/HF-Trade-App.git trading-engine
    cd trading-engine
fi

# Create virtual environment and install dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create systemd service - Phase 8 Momentum strategy
cat > /etc/systemd/system/trading-engine.service << 'EOF'
[Unit]
Description=Trading Engine - Phase 8 Momentum
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/trading-engine
Environment="PATH=/opt/trading-engine/venv/bin"
ExecStart=/opt/trading-engine/venv/bin/python -m live.runner_0dte --strategy momentum --mode paper
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Start service
systemctl daemon-reload
systemctl enable trading-engine
systemctl start trading-engine

echo "Phase 8 Momentum trading engine deployed and started!"
