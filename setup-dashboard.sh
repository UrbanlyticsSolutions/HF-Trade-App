#!/bin/bash
# Create dashboard service
cat > /etc/systemd/system/trading-dashboard.service << 'SERVICEEOF'
[Unit]
Description=Trading Dashboard
After=network.target trading-engine.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/trading-engine
Environment=PATH=/opt/trading-engine/venv/bin
ExecStart=/opt/trading-engine/venv/bin/python dashboard.py --port 8050
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICEEOF

systemctl daemon-reload
systemctl enable trading-dashboard
systemctl start trading-dashboard
echo "Dashboard service started!"
