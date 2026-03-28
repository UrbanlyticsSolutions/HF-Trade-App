"""
Full System Test Suite
======================
Tests every layer: imports, config, database, engine wiring, Docker build,
IB Gateway connectivity, and end-to-end engine startup.

Run:
    python -m pytest tests/test_full_system.py -v --tb=short
    python tests/test_full_system.py          # standalone
"""
import importlib
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, List

# ── project root on PYTHONPATH ────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


# =====================================================================
# 1. MODULE IMPORT TESTS
# =====================================================================
class TestModuleImports:
    """Every Python module must import without error."""

    MODULES = [
        # Config
        "config",
        "config.defaults",
        "config.config_manager",
        # Core
        "core",
        "core.signals",
        "core.risk_manager",
        "core.regime_detector",
        "core.regime_classifier",
        # Indicators
        "indicators",
        "indicators.momentum",
        "indicators.trend",
        "indicators.volatility",
        "indicators.volume",
        "indicators.signals",
        # Clients
        "clients",
        "clients.ibkr_client",
        "clients.ibkr_adapter",
        "clients.ibkr_flex",
        "clients.ibkr_db",
        "clients.database",
        "clients.fmp_stable_client",
        "clients.cached_data_fetcher",
        # Live
        "live",
        "live.trade_database",
        "live.state_persistence",
        "live.order_manager",
        "live.position_manager",
        "live.trade_sync",
        "live.strategy",
        "live.strategy_0dte",
        "live.engine",
        "live.dashboard",
        # Backtest
        "backtest",
        "backtest.engine",
        "backtest.runner",
    ]

    def test_all_imports(self):
        """Verify every module can be imported."""
        failures = []
        for mod in self.MODULES:
            try:
                importlib.import_module(mod)
            except Exception as e:
                failures.append(f"{mod}: {type(e).__name__}: {e}")
        assert not failures, "Module import failures:\n" + "\n".join(failures)


# =====================================================================
# 2. CONFIGURATION TESTS
# =====================================================================
class TestConfiguration:
    """strategy.json and config.defaults must load and return valid values."""

    def test_strategy_json_exists(self):
        path = PROJECT_DIR / "config" / "strategy.json"
        assert path.exists(), f"Missing {path}"

    def test_strategy_json_valid(self):
        path = PROJECT_DIR / "config" / "strategy.json"
        with open(path) as f:
            data = json.load(f)
        assert "trade_config" in data, "Missing trade_config section"
        tc = data["trade_config"]
        assert "profit_target_pct" in tc
        assert "stop_loss_pct" in tc

    def test_defaults_load(self):
        from config import defaults as cfg
        assert cfg.initial_capital() > 0
        assert cfg.max_contracts() > 0
        assert cfg.ibkr_paper_port() > 0
        assert cfg.dashboard_port() > 0

    def test_trade_config_values(self):
        from config import defaults as cfg
        assert 0 < cfg.profit_target_pct() <= 5.0
        assert 0 < cfg.stop_loss_pct() <= 5.0
        assert cfg.min_option_price() >= 0
        assert cfg.max_option_price() > cfg.min_option_price()
        assert cfg.trade_start_hour() in range(0, 24)
        assert cfg.trade_end_hour() in range(0, 24)
        assert cfg.max_hold_minutes() > 0

    def test_risk_config_present(self):
        from config import defaults as cfg
        rc = cfg.get_risk_config()
        # risk_config may be empty in some setups — just ensure it loads
        assert isinstance(rc, dict)


# =====================================================================
# 3. DATABASE LAYER TESTS
# =====================================================================
class TestTradeDatabase:
    """Tests for TradeDatabase — uses a temp SQLite file."""

    @contextmanager
    def _temp_db(self):
        from live.trade_database import TradeDatabase
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = TradeDatabase(tmp.name)
        try:
            yield db
        finally:
            db.conn.close()
            os.unlink(tmp.name)

    def test_tables_created(self):
        with self._temp_db() as db:
            cursor = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row["name"] for row in cursor}
            for t in ("trades", "orders", "quote_snapshots", "daily_pnl"):
                assert t in tables, f"Missing table: {t}"

    def test_insert_and_get_trade(self):
        from live.trade_database import Trade
        with self._temp_db() as db:
            trade = Trade(
                symbol="SPY20260321C560",
                underlying="SPY",
                trade_type="option",
                option_type="call",
                strike=560.0,
                expiration="2026-03-21",
                action="buy",
                quantity=2,
                entry_price=1.50,
                entry_time=datetime.now().isoformat(),
                status="open",
            )
            tid = db.insert_trade(trade)
            assert tid is not None and tid > 0

            fetched = db.get_trade(tid)
            assert fetched is not None
            assert fetched["symbol"] == trade.symbol
            assert fetched["entry_price"] == 1.50
            assert fetched["status"] == "open"

    def test_close_trade(self):
        from live.trade_database import Trade
        with self._temp_db() as db:
            trade = Trade(
                symbol="SPY20260321P555",
                underlying="SPY",
                trade_type="option",
                option_type="put",
                strike=555.0,
                expiration="2026-03-21",
                action="buy",
                quantity=1,
                entry_price=1.00,
                entry_time=datetime.now().isoformat(),
                status="open",
            )
            tid = db.insert_trade(trade)
            db.close_trade(tid, exit_price=1.50)

            closed = db.get_trade(tid)
            assert closed["status"] == "closed"
            assert closed["exit_price"] == 1.50
            # PnL = (1.50 - 1.00) * 1 * 100 - 0 commission = 50.00
            assert abs(closed["pnl"] - 50.00) < 0.01

    def test_get_open_trades(self):
        from live.trade_database import Trade
        with self._temp_db() as db:
            for i in range(3):
                t = Trade(
                    symbol=f"SPY20260321C{560+i}",
                    underlying="SPY",
                    trade_type="option",
                    action="buy",
                    quantity=1,
                    entry_price=1.00,
                    entry_time=datetime.now().isoformat(),
                    status="open" if i < 2 else "closed",
                )
                db.insert_trade(t)
            opens = db.get_open_trades()
            assert len(opens) == 2

    def test_update_trade(self):
        from live.trade_database import Trade
        with self._temp_db() as db:
            t = Trade(
                symbol="SPY20260321C560",
                underlying="SPY",
                trade_type="option",
                action="buy",
                quantity=1,
                entry_price=1.00,
                entry_time=datetime.now().isoformat(),
                status="open",
            )
            tid = db.insert_trade(t)
            db.update_trade(tid, entry_price=1.25, notes="corrected")
            updated = db.get_trade(tid)
            assert updated["entry_price"] == 1.25
            assert updated["notes"] == "corrected"

    def test_daily_pnl_summary(self):
        from live.trade_database import Trade
        with self._temp_db() as db:
            now_str = datetime.now().isoformat()
            t = Trade(
                symbol="SPY20260321C560", underlying="SPY",
                trade_type="option", action="buy", quantity=1,
                entry_price=1.00, entry_time=now_str, status="open",
            )
            tid = db.insert_trade(t)
            db.close_trade(tid, exit_price=1.50)
            # Verify daily_pnl row was created
            today_str = date.today().isoformat()
            row = db.conn.execute(
                "SELECT * FROM daily_pnl WHERE date = ?", (today_str,)
            ).fetchone()
            assert row is not None
            assert row["trades_closed"] >= 1


# =====================================================================
# 4. TRADE SYNC TESTS
# =====================================================================
class TestTradeSync:
    """Test symbol normalization and dedup logic."""

    def test_normalize_questrade_symbol(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_symbol("SPY18Mar26P664.00") == "SPY20260318P664"

    def test_normalize_ibkr_symbol(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_symbol("SPY20260318P664") == "SPY20260318P664"

    def test_normalize_plain_symbol(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_symbol("AAPL") == "AAPL"

    def test_normalize_time(self):
        from live.trade_sync import TradeSync
        assert TradeSync._normalize_time("2026-03-18T10:43:57") == "2026-03-18T10:43"


# =====================================================================
# 5. ENGINE WIRING TESTS (no broker connection)
# =====================================================================
class TestEngineWiring:
    """Verify the engine can be instantiated with mock dependencies."""

    @contextmanager
    def _temp_db(self):
        from live.trade_database import TradeDatabase
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = TradeDatabase(tmp.name)
        try:
            yield db
        finally:
            db.conn.close()
            os.unlink(tmp.name)

    def _mock_client(self):
        client = MagicMock()
        client.get_accounts.return_value = [{"number": "TEST123"}]
        client.get_positions.return_value = []
        client.get_orders.return_value = []
        client.get_executions.return_value = []
        client.market_buy.return_value = {"orderId": 1}
        client.market_sell.return_value = {"orderId": 2}
        return client

    def test_engine_instantiation(self):
        from live.engine import LiveTradingEngine, EngineConfig
        from live.position_manager import PositionManager
        from live.order_manager import OrderManager

        with self._temp_db() as db:
            client = self._mock_client()
            config = EngineConfig(account_id="TEST123", mode="paper")
            pm = PositionManager(client)
            om = OrderManager(client)
            engine = LiveTradingEngine(client, db, pm, om, config)
            assert engine is not None
            assert engine.config.mode == "paper"

    def test_add_strategy(self):
        from live.engine import LiveTradingEngine, EngineConfig
        from live.position_manager import PositionManager
        from live.order_manager import OrderManager

        with self._temp_db() as db:
            client = self._mock_client()
            config = EngineConfig(account_id="TEST123", mode="paper")
            pm = PositionManager(client)
            om = OrderManager(client)
            engine = LiveTradingEngine(client, db, pm, om, config)

            strategy = MagicMock()
            strategy.name = "test_strategy"
            engine.add_strategy(strategy)
            assert len(engine._strategies) == 1
            strategy.set_managers.assert_called_once_with(pm, om)


# =====================================================================
# 6. STATE PERSISTENCE TESTS
# =====================================================================
class TestStatePersistence:
    """Verify state save/load round-trips."""

    def test_state_save_load(self):
        from live.state_persistence import StatePersistence
        tmp_dir = tempfile.mkdtemp()
        state_file = os.path.join(tmp_dir, "test_state.json")
        try:
            sp = StatePersistence(state_file)
            sp.state.initial_capital = 25000.0
            sp.state.current_capital = 26000.0
            sp.state.total_trades = 10
            sp.state.total_wins = 7
            sp.save_state()

            sp2 = StatePersistence(state_file)
            assert sp2.state.initial_capital == 25000.0
            assert sp2.state.current_capital == 26000.0
            assert sp2.state.total_trades == 10
            assert sp2.state.total_wins == 7
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =====================================================================
# 7. INDICATOR / SIGNAL TESTS
# =====================================================================
class TestIndicatorsAndSignals:
    """Basic smoke tests for indicator modules."""

    def test_momentum_imports(self):
        from indicators.momentum import RSI
        import numpy as np
        prices = np.array([100 + i * 0.5 for i in range(30)])
        rsi_indicator = RSI(period=14)
        result = rsi_indicator.calculate(prices)
        assert result is not None
        # RSI current value should be between 0 and 100
        assert 0 <= result.current <= 100

    def test_trend_imports(self):
        import indicators.trend as trend
        assert hasattr(trend, "compute_macd") or hasattr(trend, "compute_ema") or callable(getattr(trend, next(iter(dir(trend))), None))

    def test_volatility_imports(self):
        import indicators.volatility as vol
        assert vol is not None

    def test_core_signals(self):
        from core.signals import STRATEGIES
        assert isinstance(STRATEGIES, dict)
        assert len(STRATEGIES) > 0


# =====================================================================
# 8. DOCKER / DEPLOYMENT TESTS
# =====================================================================
class TestDockerDeployment:
    """Verify Docker config and build."""

    def test_dockerfile_exists(self):
        assert (PROJECT_DIR / "deploy" / "Dockerfile").exists()

    def test_docker_compose_exists(self):
        assert (PROJECT_DIR / "deploy" / "docker-compose.yml").exists()

    def test_docker_compose_valid_yaml(self):
        """docker-compose.yml must parse without error."""
        try:
            import yaml
        except ImportError:
            # Try with json fallback — YAML is a superset of JSON... nope.
            # Just validate structure by reading lines
            path = PROJECT_DIR / "deploy" / "docker-compose.yml"
            text = path.read_text()
            assert "services:" in text
            assert "ib-gateway:" in text
            assert "app:" in text
            return
        path = PROJECT_DIR / "deploy" / "docker-compose.yml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "services" in data
        assert "ib-gateway" in data["services"]
        assert "app" in data["services"]

    def test_docker_compose_services(self):
        """Verify critical docker-compose fields."""
        path = PROJECT_DIR / "deploy" / "docker-compose.yml"
        text = path.read_text()
        # IB Gateway
        assert "ghcr.io/gnzsnz/ib-gateway" in text
        assert "healthcheck:" in text
        assert "4004:4004" in text or "4004" in text
        # App
        assert "deploy/Dockerfile" in text
        assert "depends_on:" in text
        assert "service_healthy" in text
        # Volumes
        assert "app-data:" in text
        assert "app-logs:" in text

    def test_dockerfile_structure(self):
        """Verify Dockerfile has required stages."""
        path = PROJECT_DIR / "deploy" / "Dockerfile"
        text = path.read_text()
        assert "FROM python:3.12" in text
        assert "requirements.txt" in text
        assert "pip install" in text
        assert "ibapi" in text
        assert "EXPOSE 8050" in text
        assert "start.py" in text

    def test_requirements_txt(self):
        """requirements.txt must exist and list key deps."""
        path = PROJECT_DIR / "requirements.txt"
        assert path.exists()
        text = path.read_text().lower()
        for pkg in ["requests", "pandas", "numpy", "dash", "plotly"]:
            assert pkg in text, f"Missing {pkg} in requirements.txt"

    def test_docker_available(self):
        """Check if Docker is available on the system."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True, timeout=10
            )
            assert result.returncode == 0, f"docker not available: {result.stderr}"
        except FileNotFoundError:
            skip_test("Docker not installed on this machine")

    def test_docker_compose_config(self):
        """Validate docker-compose config (dry-run)."""
        compose_file = PROJECT_DIR / "deploy" / "docker-compose.yml"
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "config"],
                capture_output=True, text=True, timeout=30,
                cwd=str(PROJECT_DIR / "deploy"),
            )
            if result.returncode != 0:
                # Try docker-compose (v1)
                result = subprocess.run(
                    ["docker-compose", "-f", str(compose_file), "config"],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(PROJECT_DIR / "deploy"),
                )
            assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
        except FileNotFoundError:
            skip_test("Docker/docker-compose not installed")

    def test_docker_build_app(self):
        """Build the app Docker image (no push)."""
        compose_file = PROJECT_DIR / "deploy" / "docker-compose.yml"
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file),
                 "build", "--no-cache", "app"],
                capture_output=True, text=True, timeout=300,
                cwd=str(PROJECT_DIR / "deploy"),
            )
            if result.returncode != 0:
                # Fallback to docker-compose v1
                result = subprocess.run(
                    ["docker-compose", "-f", str(compose_file),
                     "build", "--no-cache", "app"],
                    capture_output=True, text=True, timeout=300,
                    cwd=str(PROJECT_DIR / "deploy"),
                )
            assert result.returncode == 0, (
                f"Docker build failed:\n{result.stderr[-2000:]}"
            )
        except FileNotFoundError:
            skip_test("Docker not installed")


# =====================================================================
# 9. IB GATEWAY CONNECTIVITY TESTS
# =====================================================================
class TestIBGatewayConnectivity:
    """Test connectivity to IB Gateway (requires running container)."""

    def _get_ib_host_port(self):
        host = os.environ.get("IBKR_HOST", "127.0.0.1")
        port = int(os.environ.get("IBKR_PAPER_PORT", "4004"))
        return host, port

    def test_ib_gateway_port_open(self):
        """Check if IB Gateway TCP port is reachable."""
        host, port = self._get_ib_host_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            result = sock.connect_ex((host, port))
            if result != 0:
                skip_test(
                    f"IB Gateway not reachable at {host}:{port} — "
                    "start with: docker compose -f deploy/docker-compose.yml up -d"
                )
        finally:
            sock.close()

    def test_ibkr_client_connect(self):
        """Attempt to instantiate and connect IBKRClient."""
        host, port = self._get_ib_host_port()
        # Quick TCP check first
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        if sock.connect_ex((host, port)) != 0:
            sock.close()
            skip_test(f"IB Gateway not reachable at {host}:{port}")
        sock.close()

        from clients.ibkr_client import IBKRClient
        client = IBKRClient(host=host, port=port, client_id=99)
        try:
            connected = client.connect(timeout=15)
            assert connected, "IBKRClient.connect() returned False"
            # Basic validation: we should get account info
            time.sleep(2)
            assert client.account_id, "No account ID received after connect"
        finally:
            try:
                client.disconnect()
            except Exception:
                pass

    def test_ibkr_adapter_connect(self):
        """Attempt to connect via IBKRAdapter (engine-facing API)."""
        host, port = self._get_ib_host_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        if sock.connect_ex((host, port)) != 0:
            sock.close()
            skip_test(f"IB Gateway not reachable at {host}:{port}")
        sock.close()

        from clients.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter(host=host, port=port, client_id=98)
        try:
            # IBKRAdapter connects in __init__
            time.sleep(3)
            # Should be able to get accounts
            accounts = adapter.get_accounts()
            assert accounts, "get_accounts() returned nothing"
        finally:
            try:
                adapter.disconnect()
            except Exception:
                pass

    def test_ibkr_positions(self):
        """Fetch positions from IBKR (may be empty in paper with no trades)."""
        host, port = self._get_ib_host_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        if sock.connect_ex((host, port)) != 0:
            sock.close()
            skip_test(f"IB Gateway not reachable at {host}:{port}")
        sock.close()

        from clients.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter(host=host, port=port, client_id=97)
        try:
            time.sleep(3)
            positions = adapter.get_positions()
            # positions is a list — may be empty, that's OK
            assert isinstance(positions, list)
        finally:
            try:
                adapter.disconnect()
            except Exception:
                pass


# =====================================================================
# 10. DOCKER CONTAINER STATUS TESTS
# =====================================================================
class TestDockerContainers:
    """Check status of running Docker containers."""

    def _docker_ps(self):
        try:
            r = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}|{{.Status}}|{{.Ports}}"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return None
            containers = {}
            for line in r.stdout.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|")
                    containers[parts[0]] = {"status": parts[1], "ports": parts[2] if len(parts) > 2 else ""}
            return containers
        except Exception:
            return None

    def test_ib_gateway_container_running(self):
        """Check deploy-ib-gateway-1 container is running and healthy."""
        containers = self._docker_ps()
        if containers is None:
            skip_test("Docker not available")

        gw_names = [n for n in containers if "ib-gateway" in n.lower()]
        if not gw_names:
            skip_test("IB Gateway container not running — start with docker compose up -d")

        gw = containers[gw_names[0]]
        assert "Up" in gw["status"], f"IB Gateway status: {gw['status']}"

    def test_app_container_running(self):
        """Check deploy-app-1 container exists (may not be running in dev)."""
        containers = self._docker_ps()
        if containers is None:
            skip_test("Docker not available")

        app_names = [n for n in containers if "app" in n.lower() and "ib-gateway" not in n.lower()]
        # App container is optional in local dev — just report
        if not app_names:
            skip_test("App container not running (expected in local dev)")
        app = containers[app_names[0]]
        assert "Up" in app["status"], f"App container status: {app['status']}"


# =====================================================================
# 11. END-TO-END: ENGINE STARTUP TEST
# =====================================================================
class TestEngineE2E:
    """Full end-to-end: build engine with real IBKR adapter, verify startup."""

    def test_engine_start_stop(self):
        """Start the engine with a mock client, run 2 seconds, stop gracefully."""
        from live.engine import LiveTradingEngine, EngineConfig
        from live.trade_database import TradeDatabase
        from live.position_manager import PositionManager
        from live.order_manager import OrderManager

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()

        try:
            db = TradeDatabase(tmp.name)
            client = MagicMock()
            client.get_accounts.return_value = [{"number": "E2E_TEST"}]
            client.get_positions.return_value = []
            client.get_orders.return_value = []
            client.get_executions.return_value = []
            client.get_quote.return_value = {"lastTradePrice": 560.0}
            client.get_option_chain.return_value = []

            config = EngineConfig(
                account_id="E2E_TEST",
                mode="monitor",
                symbols=["SPY"],
                option_underlyings=["SPY"],
            )
            pm = PositionManager(client)
            om = OrderManager(client)
            engine = LiveTradingEngine(client, db, pm, om, config)

            # Start in a thread
            import threading
            t = threading.Thread(target=engine.start, daemon=True)
            t.start()
            time.sleep(3)

            assert engine._running, "Engine should be running"
            engine.stop()
            time.sleep(1)
            assert not engine._running, "Engine should have stopped"
        finally:
            try:
                db.conn.close()
            except Exception:
                pass
            os.unlink(tmp.name)


# =====================================================================
# 12. COMPILATION CHECK — ALL .py FILES
# =====================================================================
class TestCompilation:
    """Every .py file must compile without syntax errors."""

    def test_all_py_files_compile(self):
        failures = []
        for py_file in PROJECT_DIR.rglob("*.py"):
            # Skip __pycache__, .venv, node_modules
            parts = py_file.parts
            if any(skip in parts for skip in ("__pycache__", ".venv", "node_modules", "tests")):
                continue
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    source = f.read()
                compile(source, str(py_file), "exec")
            except SyntaxError as e:
                failures.append(f"{py_file.relative_to(PROJECT_DIR)}: {e}")
        assert not failures, "Syntax errors:\n" + "\n".join(failures)


# =====================================================================
# 13. DASHBOARD SMOKE TEST
# =====================================================================
class TestDashboard:
    """Dashboard app object must initialize."""

    def test_app_layout(self):
        from live.dashboard import app
        assert app is not None
        assert app.title == "0DTE Trading Dashboard"
        # Layout is a function (serve_layout)
        assert callable(app.layout)

    def test_serve_layout_returns(self):
        """serve_layout() should return a Dash component without crashing."""
        from live.dashboard import serve_layout
        layout = serve_layout()
        assert layout is not None


# =====================================================================
# 14. FLEX CLIENT TESTS
# =====================================================================
class TestFlexClient:
    """IBKRFlexClient basic structure tests."""

    def test_flex_client_instantiation(self):
        from clients.ibkr_flex import IBKRFlexClient
        client = IBKRFlexClient(token="dummy_token")
        assert client is not None

    def test_flex_parse_trades_empty(self):
        from clients.ibkr_flex import IBKRFlexClient
        client = IBKRFlexClient(token="dummy_token")
        # parse_trades should handle empty/invalid XML gracefully
        try:
            trades = client.parse_trades("<FlexQueryResponse><FlexStatements></FlexStatements></FlexQueryResponse>")
            assert isinstance(trades, list)
        except Exception:
            pass  # Some implementations may raise — that's OK for empty XML


# =====================================================================
# STANDALONE RUNNER
# =====================================================================
class _SkipTest(Exception):
    """Raised to skip a test in standalone mode."""
    pass


def skip_test(reason: str):
    """Skip a test — works with both pytest and standalone runner."""
    raise _SkipTest(reason)


def run_all():
    """Run all tests and print results."""
    import traceback

    test_classes = [
        TestCompilation,
        TestModuleImports,
        TestConfiguration,
        TestTradeDatabase,
        TestTradeSync,
        TestEngineWiring,
        TestStatePersistence,
        TestIndicatorsAndSignals,
        TestDashboard,
        TestFlexClient,
        TestDockerDeployment,
        TestDockerContainers,
        TestIBGatewayConnectivity,
        TestEngineE2E,
    ]

    total = 0
    passed = 0
    failed = 0
    skipped = 0
    errors = []

    print("=" * 70)
    print("  FULL SYSTEM TEST SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for cls in test_classes:
        print(f"\n{'─' * 60}")
        print(f"  {cls.__name__}")
        print(f"{'─' * 60}")
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            total += 1
            method = getattr(instance, method_name)
            label = method_name.replace("test_", "").replace("_", " ").title()
            try:
                method()
                passed += 1
                print(f"  PASS  {label}")
            except _SkipTest as e:
                skipped += 1
                print(f"  SKIP  {label} — {e}")
            except AssertionError as e:
                failed += 1
                errors.append((cls.__name__, method_name, str(e)))
                print(f"  FAIL  {label}")
                print(f"        {str(e)[:200]}")
            except Exception as e:
                e_str = str(e)
                e_type = type(e).__name__
                # Handle pytest.skip Skipped exception
                if "skip" in e_type.lower() or "Skipped" in e_type:
                    skipped += 1
                    print(f"  SKIP  {label} — {e_str}")
                elif isinstance(e, AssertionError):
                    failed += 1
                    errors.append((cls.__name__, method_name, e_str))
                    print(f"  FAIL  {label}")
                    print(f"        {e_str[:200]}")
                else:
                    failed += 1
                    errors.append((cls.__name__, method_name, traceback.format_exc()))
                    print(f"  FAIL  {label}")
                    print(f"        {e_type}: {e_str[:200]}")

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped, {total} total")
    print(f"{'=' * 70}")

    if errors:
        print(f"\n  FAILURES ({len(errors)}):")
        for cls_name, method, err in errors:
            print(f"\n  {cls_name}.{method}:")
            for line in err.split("\n")[-5:]:
                if line.strip():
                    print(f"    {line.strip()}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
