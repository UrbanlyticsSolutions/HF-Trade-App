"""
0DTE Paper Trading Launcher

Starts both the trading engine and dashboard in parallel.

Usage:
    python start.py                    # Start paper trading + dashboard
    python start.py --mode monitor     # Monitor only + dashboard
    python start.py --mode live        # Live trading + dashboard
    python start.py --dashboard-only   # Dashboard only
    python start.py --engine-only      # Engine only
"""
import subprocess
import sys
import os
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime
import threading

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent


def run_engine(mode: str = "paper", capital: float = None, verbose: bool = False):
    """Run the trading engine in a subprocess."""
    cmd = [
        sys.executable, "-m", "live.runner_0dte",
        "--mode", mode,
    ]
    if capital is not None:
        cmd.extend(["--capital", str(capital)])
    if verbose:
        cmd.append("-v")
    
    print(f"[ENGINE] Starting 0DTE trading engine (mode={mode})...")
    return subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )


def run_dashboard(port: int = 8050):
    """Run the Plotly dashboard in a subprocess."""
    cmd = [sys.executable, str(SCRIPT_DIR / "live" / "dashboard.py"), "--port", str(port)]
    
    print(f"[DASHBOARD] Starting dashboard on http://localhost:{port}...")
    return subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )


# Shared output file for dashboard to read
OUTPUT_FILE = SCRIPT_DIR / "logs" / "terminal_output.log"
MAX_OUTPUT_LINES = 500  # Keep last 500 lines

def write_to_output_file(line: str):
    """Write line to shared output file that dashboard can read."""
    try:
        # Ensure logs directory exists
        OUTPUT_FILE.parent.mkdir(exist_ok=True)
        
        # Read existing lines
        existing = []
        if OUTPUT_FILE.exists():
            with open(OUTPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                existing = f.readlines()
        
        # Add new line with timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        new_line = f"[{timestamp}] {line}\n"
        existing.append(new_line)
        
        # Keep only last N lines
        if len(existing) > MAX_OUTPUT_LINES:
            existing = existing[-MAX_OUTPUT_LINES:]
        
        # Write back
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.writelines(existing)
    except:
        pass


def stream_output(process, prefix: str):
    """Stream output from a subprocess."""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                if line:
                    # Write ALL lines to file for dashboard
                    write_to_output_file(f"[{prefix}] {line}")
                    
                    # Print important lines to console
                    if any(keyword in line for keyword in [
                        'ENTRY', 'EXIT', 'Signal', 'ERROR', 'WARNING', 
                        'Successfully', 'Starting', 'Connecting', 'P&L',
                        'TRADING', 'Dash is running', 'http://', 'Got',
                        'options for', 'token', 'Fetching',
                        'Order', 'order', 'placeOrder', 'resolve_contract',
                        'conId', 'reqContract'
                    ]):
                        print(f"[{prefix}] {line}")
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description="0DTE Paper Trading Launcher")
    parser.add_argument("--mode", default="paper", choices=["monitor", "paper", "live"],
                       help="Trading mode (default: paper)")
    parser.add_argument("--capital", type=float, default=None, help="Override capital (default: auto from broker)")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--dashboard-only", action="store_true", help="Run dashboard only")
    parser.add_argument("--engine-only", action="store_true", help="Run engine only")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  0DTE PAPER TRADING SYSTEM")
    print("=" * 60)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Capital: {'$' + f'{args.capital:,.0f}' if args.capital else 'auto (from broker)'}")
    print(f"  Dashboard: http://localhost:{args.port}")
    print("=" * 60)
    print()
    
    # Clear terminal output file on startup
    try:
        OUTPUT_FILE.parent.mkdir(exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] [SYSTEM] Starting 0DTE Paper Trading System...\n")
    except:
        pass
    
    processes = []
    threads = []
    
    try:
        # Start engine (unless dashboard-only)
        if not args.dashboard_only:
            engine_proc = run_engine(args.mode, args.capital, args.verbose)
            processes.append(("ENGINE", engine_proc))
            
            # Stream engine output in background thread
            t = threading.Thread(target=stream_output, args=(engine_proc, "ENGINE"), daemon=True)
            t.start()
            threads.append(t)
            
            # Give engine time to connect
            time.sleep(3)
        
        # Start dashboard (unless engine-only)
        if not args.engine_only:
            dash_proc = run_dashboard(args.port)
            processes.append(("DASHBOARD", dash_proc))
            
            # Stream dashboard output in background thread
            t = threading.Thread(target=stream_output, args=(dash_proc, "DASHBOARD"), daemon=True)
            t.start()
            threads.append(t)
            # Browser is opened once by dashboard.py main(); do not call webbrowser here
            # or two tabs will open.
        
        print("\n[SYSTEM] All services started. Press Ctrl+C to stop.\n")
        
        # Restart signal file
        restart_signal_file = SCRIPT_DIR / "logs" / "restart_signal.txt"
        
        # Wait for processes
        while True:
            time.sleep(1)
            
            # Check for restart signal from dashboard
            if restart_signal_file.exists():
                print("\n[SYSTEM] Restart signal received from dashboard!")
                write_to_output_file("[SYSTEM] Restarting engine...")
                
                # Delete signal file
                try:
                    restart_signal_file.unlink()
                except:
                    pass
                
                # Kill and restart engine
                for name, proc in processes:
                    if name == "ENGINE":
                        print(f"[{name}] Stopping for restart...")
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        
                        # Remove from list
                        processes = [(n, p) for n, p in processes if n != "ENGINE"]
                        
                        # Wait a moment
                        time.sleep(2)
                        
                        # Start new engine
                        print("[ENGINE] Restarting...")
                        write_to_output_file("[SYSTEM] Engine restarted!")
                        engine_proc = run_engine(args.mode, args.capital, args.verbose)
                        processes.append(("ENGINE", engine_proc))
                        
                        t = threading.Thread(target=stream_output, args=(engine_proc, "ENGINE"), daemon=True)
                        t.start()
                        threads.append(t)
                        
                        print("[SYSTEM] Engine restarted successfully!")
                        break
            
            # Check if any process died
            for name, proc in list(processes):
                if proc.poll() is not None:
                    print(f"\n[{name}] Process exited with code {proc.returncode}")
                    if name == "ENGINE":
                        write_to_output_file(f"[SYSTEM] Engine exited (code {proc.returncode}), restarting in 30s...")
                        processes = [(n, p) for n, p in processes if n != "ENGINE"]
                        time.sleep(30)
                        print("[ENGINE] Auto-restarting...")
                        write_to_output_file("[SYSTEM] Engine auto-restarting...")
                        engine_proc = run_engine(args.mode, args.capital, args.verbose)
                        processes.append(("ENGINE", engine_proc))
                        t = threading.Thread(target=stream_output, args=(engine_proc, "ENGINE"), daemon=True)
                        t.start()
                        threads.append(t)
                    else:
                        raise KeyboardInterrupt
                    
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down...")
        
        for name, proc in processes:
            print(f"[{name}] Stopping...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        
        print("[SYSTEM] All services stopped.")


if __name__ == "__main__":
    main()
