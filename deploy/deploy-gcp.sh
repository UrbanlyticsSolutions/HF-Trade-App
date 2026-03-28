#!/bin/bash
###############################################################################
# deploy-gcp.sh — Deploy IB Gateway + Trading App to a GCP Compute Engine VM
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated: gcloud auth login
#   2. A GCP project selected: gcloud config set project YOUR_PROJECT
#   3. Fill in deploy/.env with your IB credentials
#
# Usage:
#   bash deploy/deploy-gcp.sh              # Create VM + deploy
#   bash deploy/deploy-gcp.sh --teardown   # Delete VM
###############################################################################
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
VM_NAME="ib-trading"
ZONE="us-east1-b"             # Low latency to IB servers (NY/Chicago)
MACHINE_TYPE="e2-small"       # 2 vCPU, 2GB RAM — sufficient for IB Gateway + Python
DISK_SIZE="20GB"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
FIREWALL_RULE="allow-trading-dashboard"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Teardown ──────────────────────────────────────────────────
if [[ "${1:-}" == "--teardown" ]]; then
    echo "Deleting VM $VM_NAME..."
    gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet || true
    gcloud compute firewall-rules delete "$FIREWALL_RULE" --quiet || true
    echo "Done."
    exit 0
fi

# ── Create Firewall Rule (dashboard + noVNC) ─────────────────
echo "==> Creating firewall rule..."
gcloud compute firewall-rules create "$FIREWALL_RULE" \
    --allow=tcp:8050,tcp:6080 \
    --target-tags=trading-vm \
    --description="Allow dashboard (8050) and noVNC (6080)" \
    2>/dev/null || echo "  (firewall rule already exists)"

# ── Create VM ────────────────────────────────────────────────
echo "==> Creating VM: $VM_NAME ($MACHINE_TYPE in $ZONE)..."
gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$DISK_SIZE" \
    --tags=trading-vm \
    --metadata=startup-script='#!/bin/bash
# Install Docker
curl -fsSL https://get.docker.com | sh
usermod -aG docker $USER
systemctl enable docker
# Install Docker Compose plugin
apt-get install -y docker-compose-plugin
'

echo "==> Waiting for VM to be ready..."
sleep 30

# ── Upload project files ─────────────────────────────────────
echo "==> Uploading project to VM..."
# Create a tarball excluding unnecessary files
cd "$PROJECT_DIR"
tar czf /tmp/trading-app.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='output' \
    --exclude='.git' \
    --exclude='*.db' \
    --exclude='*.db-journal' \
    .

gcloud compute scp /tmp/trading-app.tar.gz "$VM_NAME":~ --zone="$ZONE"
rm /tmp/trading-app.tar.gz

# ── Deploy on VM ──────────────────────────────────────────────
echo "==> Deploying on VM..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command='
    mkdir -p ~/trading-app
    cd ~/trading-app
    tar xzf ~/trading-app.tar.gz
    rm ~/trading-app.tar.gz

    # Copy .dockerignore to project root for Docker build context
    cp deploy/.dockerignore .dockerignore

    # Wait for Docker to be ready
    for i in $(seq 1 30); do
        docker info >/dev/null 2>&1 && break
        echo "  Waiting for Docker..."
        sleep 5
    done

    # Start services
    cd deploy
    docker compose up -d --build

    echo ""
    echo "==> Deployment complete!"
    docker compose ps
'

# ── Print access info ─────────────────────────────────────────
EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "============================================"
echo "  DEPLOYMENT COMPLETE"
echo "============================================"
echo "  VM:        $VM_NAME ($ZONE)"
echo "  Dashboard: http://$EXTERNAL_IP:8050"
echo "  noVNC:     http://$EXTERNAL_IP:6080"
echo "  SSH:       gcloud compute ssh $VM_NAME --zone=$ZONE"
echo "============================================"
echo ""
echo "NEXT STEPS:"
echo "  1. Open noVNC (http://$EXTERNAL_IP:6080) to complete IB login/2FA"
echo "  2. Check logs: gcloud compute ssh $VM_NAME --zone=$ZONE --command='cd ~/trading-app/deploy && docker compose logs -f app'"
