# Workflow

## Setup (one-time): Flex Web Service for trade history sync

The engine syncs IBKR trade history at startup. For total PnL and historical trades to work (especially after TWS restarts), configure the Flex Web Service:

1. **Create an Activity Flex Query**
   - IBKR Account Management → Reports → Flex Queries
   - Create a new **Activity Flex Query** with **Trades** enabled, XML format
   - Save and note the numeric **Query ID**

2. **Generate Flex token**
   - Same area → **Configure Flex Web Service**
   - Enable Flex Web Service, then **Generate New Token**
   - Copy the token (shown once)

3. **Set environment variables** (in `.env` or your shell)
   ```
   IBKR_FLEX_TOKEN=<your-token>
   IBKR_FLEX_QUERY_ID=<query-id>
   ```

4. **Verify**
   ```bash
   python clients/ibkr_flex.py
   ```

Without these, sync falls back to session executions only (empty after gateway restarts).

**Startup order:** engine runs `TradeSync.sync_all()` first (Flex → session fills → external DBs/CSV), then position sync, so Flex is part of the sync/position build plan.

---

## Local

```bash
python -m backtest.run backtest           # Backtest
python -m pytest tests/ -v               # Tests
python start.py --mode paper              # Paper trade (TWS on localhost:7497)
```

## Package (validate + build)

```bash
python scripts/deploy.py package          # Backtest + tests + docker build
python scripts/deploy.py package --up     # + docker compose up -d
python scripts/deploy.py package --skip-backtest --skip-tests
```

## Docker

```bash
cd deploy && docker compose up -d --build
# Dashboard: http://localhost:8040
# IB login: VNC localhost:6060
```

Ensure `IBKR_FLEX_TOKEN` and `IBKR_FLEX_QUERY_ID` are in `deploy/.env` (or your compose env) for trade history sync.

## GCP (requires gcloud CLI)

```bash
python scripts/deploy.py deploy          # Full deploy: create VM + upload + start
python scripts/deploy.py sync            # Sync code to existing VM
python scripts/deploy.py teardown        # Delete VM
```
