#!/usr/bin/env python3
"""
Deploy workflow — all Python, no bash.

  python scripts/deploy.py package         # Backtest + tests + docker build
  python scripts/deploy.py package --up    # + docker compose up -d
  python scripts/deploy.py deploy          # Full GCP: create VM + upload + start
  python scripts/deploy.py sync            # Sync code to existing GCP VM
  python scripts/deploy.py teardown        # Delete GCP VM
"""

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VM_NAME = "ib-trading"
ZONE = "us-east1-b"
REMOTE = "~/trading-app"


def run(cmd: list[str], cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=cwd)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r


def gcloud(*args: str) -> None:
    run(["gcloud", *args])


def create_tarball() -> Path:
    exclude = {"output", "logs", "__pycache__", ".git", ".env"}

    def skip(p: Path) -> bool:
        parts = p.parts
        return any(x in parts for x in exclude) or p.suffix in (".pyc", ".pyo") or p.name.endswith(".db")

    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    tmp.close()
    with tarfile.open(tmp.name, "w:gz") as tar:
        for f in ROOT.rglob("*"):
            if f.is_file() and not skip(f.relative_to(ROOT)):
                tar.add(f, arcname=f.relative_to(ROOT))
    return Path(tmp.name)


def cmd_package(args: argparse.Namespace) -> None:
    if not args.skip_backtest:
        print("\n==> Backtest")
        cmd = [sys.executable, "-m", "backtest.run", "backtest"]
        if args.year:
            cmd.extend(["--year", str(args.year)])
        run(cmd)

    if not args.skip_tests:
        print("\n==> Tests")
        run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])

    print("\n==> Docker build")
    run(["docker", "compose", "build", "--no-cache"], cwd=ROOT / "deploy")

    if args.up:
        print("\n==> Start")
        run(["docker", "compose", "up", "-d"], cwd=ROOT / "deploy")
        print("Dashboard: http://localhost:8040")
    print("\nDone.")


def cmd_deploy(args: argparse.Namespace) -> None:
    print("\n==> Firewall")
    run(["gcloud", "compute", "firewall-rules", "create", "allow-trading-dashboard",
         "--allow=tcp:8050,tcp:6080", "--target-tags=trading-vm",
         "--description=Allow dashboard and noVNC"], check=False)  # may exist

    print("\n==> Create VM")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write("""#!/bin/bash
curl -fsSL https://get.docker.com | sh
usermod -aG docker $USER
systemctl enable docker
apt-get install -y docker-compose-plugin
""")
        startup_file = f.name
    try:
        gcloud("compute", "instances", "create", VM_NAME,
               f"--zone={ZONE}", "--machine-type=e2-small", "--image-family=ubuntu-2204-lts",
               "--image-project=ubuntu-os-cloud", "--boot-disk-size=20GB", "--tags=trading-vm",
               f"--metadata-from-file=startup-script={startup_file}")
    finally:
        Path(startup_file).unlink(missing_ok=True)

    print("\n==> Wait for SSH...")
    import time
    time.sleep(30)

    tarball = create_tarball()
    try:
        print("\n==> Upload")
        gcloud("compute", "scp", str(tarball), f"{VM_NAME}:~/trading-app.tar.gz", f"--zone={ZONE}")

        print("\n==> Extract and start")
        cmd = (f"mkdir -p {REMOTE} && cd {REMOTE} && tar xzf ~/trading-app.tar.gz && rm ~/trading-app.tar.gz && "
               "cp deploy/.dockerignore .dockerignore 2>/dev/null || true && "
               "for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 5; done && "
               "cd deploy && docker compose up -d --build && docker compose ps")
        gcloud("compute", "ssh", VM_NAME, f"--zone={ZONE}", f"--command={cmd}")
    finally:
        tarball.unlink(missing_ok=True)

    ip = subprocess.run(["gcloud", "compute", "instances", "describe", VM_NAME, f"--zone={ZONE}",
                         "--format=get(networkInterfaces[0].accessConfigs[0].natIP)"],
                        capture_output=True, text=True, check=True).stdout.strip()
    print(f"\nDashboard: http://{ip}:8050  |  noVNC: http://{ip}:6080")


def cmd_sync(args: argparse.Namespace) -> None:
    tarball = create_tarball()
    try:
        print("\n==> Upload")
        gcloud("compute", "scp", str(tarball), f"{VM_NAME}:~/trading-app.tar.gz", f"--zone={ZONE}")

        print("\n==> Extract and restart")
        cmd = (f"mkdir -p {REMOTE} && cd {REMOTE} && tar xzf ~/trading-app.tar.gz && rm ~/trading-app.tar.gz && "
               "cp deploy/.dockerignore .dockerignore 2>/dev/null || true && "
               "cd deploy && docker compose up -d --build && docker compose ps")
        gcloud("compute", "ssh", VM_NAME, f"--zone={ZONE}", f"--command={cmd}")
    finally:
        tarball.unlink(missing_ok=True)
    print("\nDone.")


def cmd_teardown(args: argparse.Namespace) -> None:
    if args.yes or input(f"Delete VM {VM_NAME}? (y/N): ").strip().lower() == "y":
        print("\n==> Delete VM")
        run(["gcloud", "compute", "instances", "delete", VM_NAME, f"--zone={ZONE}", "--quiet"], check=False)
        print("\n==> Delete firewall")
        run(["gcloud", "compute", "firewall-rules", "delete", "allow-trading-dashboard", "--quiet"], check=False)
        print("Done.")
    else:
        print("Aborted.")


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy workflow")
    sub = p.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("package")
    pk.add_argument("--skip-backtest", action="store_true")
    pk.add_argument("--skip-tests", action="store_true")
    pk.add_argument("--up", action="store_true")
    pk.add_argument("--year", type=int)
    pk.set_defaults(handler=cmd_package)

    sub.add_parser("deploy").set_defaults(handler=cmd_deploy)
    sub.add_parser("sync").set_defaults(handler=cmd_sync)

    td = sub.add_parser("teardown")
    td.add_argument("-y", "--yes", action="store_true")
    td.set_defaults(handler=cmd_teardown)

    args = p.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
