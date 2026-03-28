# Trading System Refactoring Tasks

## Project: Database & Code Efficiency Overhaul

**Started:** 2026-03-28  
**Goal:** Refactor database layer to respect IBKR Flex as source of truth, eliminate code duplication, and implement comprehensive tests.

---

## Critical Issues Identified

### 1. Database Layer Problems
- **Duplicate table creation** in `clients/database.py` (lines 80-93 and 139-155)
- **Multiple database schemas** across files (`database.py`, `ibkr_db.py`, `cached_data_fetcher.py`)
- **No connection pooling** - each file opens its own connection
- **Inconsistent table structures** - similar data stored differently

### 2. IBKR Flex Integration
- **Flex is source of truth** for historical trades (survives TWS restarts)
- **TradeSync** already handles Flex imports correctly
- **Database schema should align** with Flex XML structure
- **Test coverage missing** for Flex import paths

### 3. Code Organization Issues
- **2,166 lines** in `live/engine.py` (should be <500)
- **95 Python files** with mixed responsibilities
- **50+ scripts** in root `scripts/` folder
- **No dependency injection** - tight coupling

---

## Task Breakdown

### Phase 1: Database Consolidation (PRIORITY: HIGH)

#### Task 1.1: Fix Duplicate Table Creation
**File:** `clients/database.py`  
**Status:** PENDING  
**Description:** Remove duplicate `intraday_ticker_data` table definition

**Actions:**
- [ ] Remove lines 139-155 (duplicate table creation)
- [ ] Verify no other duplicates exist
- [ ] Run existing tests to ensure no breakage

**Acceptance Criteria:**
- No duplicate CREATE TABLE statements
- All existing tests pass
- Database initialization completes without errors

---

#### Task 1.2: Consolidate Database Schemas
**Files:** `clients/database.py`, `clients/ibkr_db.py`, `clients/cached_data_fetcher.py`  
**Status:** PENDING  
**Description:** Create unified database schema that respects IBKR Flex structure

**Actions:**
- [ ] Analyze IBKR Flex XML structure (from `ibkr_flex.py`)
- [ ] Design unified schema based on Flex fields:
  - `symbol` / `underlyingSymbol`
  - `dateTime` / `tradeDate`
  - `quantity`, `tradePrice`, `proceeds`
  - `buySell`, `assetCategory`
  - `putCall`, `strike`, `expiry`
  - `fifoPnlRealized`, `ibCommission`
  - `ibExecID`, `ibOrderID`
- [ ] Create migration script to consolidate tables
- [ ] Update all clients to use unified schema

**Acceptance Criteria:**
- Single source of truth for database schema
- All Flex fields map to database columns
- No information loss during migration
- Existing data preserved

---

#### Task 1.3: Implement Connection Pooling
**Status:** PENDING  
**Description:** Replace individual connections with pooled connections

**Actions:**
- [ ] Create `DatabaseConnectionPool` class
- [ ] Implement thread-safe connection management
- [ ] Replace all `sqlite3.connect()` calls with pool
- [ ] Add connection lifecycle management

**Acceptance Criteria:**
- Single connection pool instance
- Thread-safe operations
- Reduced connection overhead
- No connection leaks

---

### Phase 2: Test Implementation (PRIORITY: HIGH)

#### Task 2.1: Database Tests Based on IBKR Flex
**File:** `tests/test_ibkr_flex_db.py` (NEW)  
**Status:** IN PROGRESS  
**Description:** Comprehensive tests for database operations with Flex data

**Test Cases:**
- [ ] Test Flex XML parsing
- [ ] Test trade insertion from Flex data
- [ ] Test duplicate detection (exec_id based)
- [ ] Test symbol normalization (IBKR vs Questrade formats)
- [ ] Test time normalization
- [ ] Test PnL calculation matches Flex
- [ ] Test commission tracking
- [ ] Test option contract parsing (calls/puts)
- [ ] Test expiry/strike extraction
- [ ] Test realized vs unrealized PnL

**Acceptance Criteria:**
- 100% coverage of Flex import path
- All edge cases tested
- Tests pass with real Flex data samples
- Performance tests for bulk inserts

---

#### Task 2.2: TradeSync Tests
**File:** `tests/test_trade_sync_comprehensive.py` (NEW)  
**Status:** PENDING  
**Description:** Comprehensive tests for trade synchronization

**Test Cases:**
- [ ] Test IBKR session execution sync
- [ ] Test Flex Web Service sync
- [ ] Test external DB sync
- [ ] Test CSV import
- [ ] Test full sync (all sources)
- [ ] Test deduplication logic
- [ ] Test FIFO pairing
- [ ] Test multi-day trade matching
- [ ] Test option vs stock handling
- [ ] Test error handling (missing fields, invalid data)

**Acceptance Criteria:**
- 100% coverage of TradeSync methods
- Tests handle all data sources
- Deduplication verified
- Error cases covered

---

#### Task 2.3: Database Integrity Tests
**File:** `tests/test_database_integrity.py` (NEW)  
**Status:** PENDING  
**Description:** Test database schema and constraints

**Test Cases:**
- [ ] Test table creation
- [ ] Test index creation
- [ ] Test foreign key constraints (if any)
- [ ] Test unique constraints
- [ ] Test data types
- [ ] Test nullable columns
- [ ] Test default values
- [ ] Test bulk insert performance
- [ ] Test concurrent access
- [ ] Test transaction rollback

**Acceptance Criteria:**
- Schema matches design
- All constraints enforced
- No data corruption
- Handles concurrent access

---

### Phase 3: Code Refactoring (PRIORITY: MEDIUM)

#### Task 3.1: Modularize live/engine.py
**Status:** PENDING  
**Description:** Break down 2,166-line engine into smaller modules

**Target Structure:**
```
live/
├── engine.py (orchestration only, <300 lines)
├── market_data_manager.py
├── trade_coordinator.py
├── execution_engine.py
└── health_monitor.py
```

**Actions:**
- [ ] Extract market data fetching logic
- [ ] Extract order execution logic
- [ ] Extract position tracking logic
- [ ] Extract health reporting logic
- [ ] Create clean interfaces between modules
- [ ] Update imports across codebase

**Acceptance Criteria:**
- Each module <500 lines
- Clear separation of concerns
- No circular dependencies
- All existing functionality preserved

---

#### Task 3.2: Create Unified Broker Interface
**Status:** PENDING  
**Description:** Standardize broker client interface using Adapter pattern

**Actions:**
- [ ] Define `BrokerClient` abstract base class
- [ ] Methods: `get_quotes()`, `place_order()`, `get_positions()`, etc.
- [ ] Refactor IBKRAdapter to implement interface
- [ ] Refactor QuestradeClient to implement interface
- [ ] Refactor WealthsimpleClient to implement interface
- [ ] Update engine to use interface

**Acceptance Criteria:**
- Single interface for all brokers
- Easy to add new brokers
- Consistent error handling
- Type-safe operations

---

#### Task 3.3: Organize Scripts Directory
**Status:** PENDING  
**Description:** Organize 50+ scripts into logical subdirectories

**Target Structure:**
```
scripts/
├── analyze/      (existing, move analysis scripts)
├── backtest/     (existing, move backtest scripts)
├── optimize/     (existing, move optimization scripts)
├── deploy/       (existing, move deployment scripts)
├── maintenance/  (NEW: cleanup, fix, verify scripts)
├── data/         (NEW: sync, query, import scripts)
└── utils/        (NEW: calc_pnl, get_positions, etc.)
```

**Actions:**
- [ ] Categorize all scripts
- [ ] Create new subdirectories
- [ ] Move scripts to appropriate locations
- [ ] Update imports and references
- [ ] Create scripts/README.md with usage guide

**Acceptance Criteria:**
- All scripts categorized
- No scripts in root scripts/ folder
- Documentation for each category
- Easy to find scripts by purpose

---

### Phase 4: Configuration & Error Handling (PRIORITY: MEDIUM)

#### Task 4.1: Centralize Configuration
**Status:** PENDING  
**Description:** Create unified configuration management

**Actions:**
- [ ] Create `ConfigManager` class
- [ ] Load from: JSON, env vars, defaults (priority order)
- [ ] Validate configuration on startup
- [ ] Provide type-safe access methods
- [ ] Remove hardcoded values across codebase

**Acceptance Criteria:**
- Single source of truth for config
- Type-safe configuration access
- Validation prevents invalid states
- Easy to override for testing

---

#### Task 4.2: Implement Centralized Error Handling
**Status:** PENDING  
**Description:** Create unified error handling and logging

**Actions:**
- [ ] Create custom exception hierarchy
- [ ] Implement error middleware
- [ ] Standardize logging format
- [ ] Add structured logging (JSON)
- [ ] Create error reporting system

**Acceptance Criteria:**
- Consistent error handling
- All exceptions logged with context
- Easy to debug issues
- Alerts for critical errors

---

## Progress Tracking

### Completed Tasks
*None yet*

### Completed Tasks
*None yet*

### Blocked Tasks
*None currently*

---

## Testing Strategy

### Unit Tests
- Database operations
- TradeSync logic
- Signal generation
- Configuration management

### Integration Tests
- Full trade flow (entry → exit)
- Multi-source sync
- Broker client interactions
- Dashboard updates

### Performance Tests
- Bulk data inserts
- Concurrent database access
- Real-time quote processing
- Large portfolio handling

---

## Risk Assessment

### High Risk
1. **Data loss during migration** - Mitigation: comprehensive backups, migration testing
2. **Breaking existing functionality** - Mitigation: extensive test coverage before refactoring

### Medium Risk
1. **Performance degradation** - Mitigation: benchmark before/after, connection pooling
2. **Broker API changes** - Mitigation: abstraction layer, version pinning

### Low Risk
1. **Script organization** - Mitigation: keep old paths as symlinks temporarily
2. **Configuration changes** - Mitigation: backward compatibility layer

---

## Success Metrics

### Code Quality
- [ ] Max file size <500 lines
- [ ] Test coverage >80%
- [ ] Zero duplicate code
- [ ] All functions <50 lines

### Performance
- [ ] Database queries <10ms average
- [ ] Bulk inserts >10,000 rows/sec
- [ ] Connection pool utilization >90%
- [ ] Zero connection leaks

### Maintainability
- [ ] Clear module boundaries
- [ ] Comprehensive documentation
- [ ] Easy to add new features
- [ ] Simple deployment process

---

## Next Actions

1. **IMMEDIATE:** Fix duplicate table creation in `clients/database.py`
2. **NEXT:** Implement comprehensive database tests
3. **THEN:** Consolidate database schemas
4. **THEN:** Refactor engine.py

---

## Notes

- **IBKR Flex is the source of truth** for historical trade data
- **TradeSync** already correctly handles Flex imports
- **Database schema** should align with Flex XML structure
- **No mock data** - tests should use real Flex data samples
- **Fail fast** - better to error than use incorrect data

---

## References

- IBKR Flex Web Service: `clients/ibkr_flex.py`
- Trade Sync Logic: `live/trade_sync.py`
- Current Database: `clients/database.py`
- IBKR Database: `clients/ibkr_db.py`
- Workflow Documentation: `WORKFLOW.md`
