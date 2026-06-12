CREATE TABLE IF NOT EXISTS cycles (
    cycle_id TEXT PRIMARY KEY,
    start_ts TEXT NOT NULL,
    end_ts TEXT,
    status TEXT,
    error_msg TEXT,
    git_commit_sha TEXT,
    n_trades INTEGER,
    notes TEXT,
    critical_data_fail_sources TEXT,
    supplementary_stale_sources TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    coin TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    model_path_sha TEXT,
    pred_value REAL,
    pred_quantile_low REAL,
    pred_quantile_high REAL,
    ref_price REAL,
    signal_h7 INTEGER,
    signal_h14 INTEGER,
    consensus_signal INTEGER,
    bundle_route TEXT
);

CREATE TABLE IF NOT EXISTS sizing (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    coin TEXT NOT NULL,
    realized_vol REAL,
    target_vol REAL,
    kelly REAL,
    confidence REAL,
    base_size REAL,
    leverage REAL,
    sma30_multiplier REAL,
    final_size_notional REAL
);

CREATE TABLE IF NOT EXISTS risk_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    coin TEXT,
    check_name TEXT NOT NULL,
    passed INTEGER NOT NULL,
    value REAL,
    threshold REAL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    coin TEXT NOT NULL,
    side TEXT,
    qty REAL,
    entry_price REAL,
    exit_price REAL,
    pnl REAL,
    fees REAL,
    slippage REAL,
    order_id TEXT,
    stop_loss_id TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    total_value REAL,
    usdt_balance REAL,
    position_qty_per_coin TEXT,
    unrealized_pnl REAL
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    coin TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value REAL,
    source TEXT
);

CREATE TABLE IF NOT EXISTS model_artifacts (
    retrain_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    model_path TEXT NOT NULL,
    train_window_start TEXT,
    train_window_end TEXT,
    train_rows INTEGER,
    train_dir_acc_h7 REAL,
    train_dir_acc_h14 REAL,
    sha256 TEXT
);

CREATE TABLE IF NOT EXISTS retrains (
    retrain_id TEXT PRIMARY KEY,
    cycle_id TEXT,
    checkpoint_path TEXT,
    checkpoint_sha TEXT,
    n_train_rows INTEGER,
    train_window_start TEXT,
    train_dir_acc REAL,
    status TEXT,
    routes TEXT
);

CREATE TABLE IF NOT EXISTS shadow_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    coin TEXT NOT NULL,
    live_signal INTEGER,
    backtest_signal INTEGER,
    agree INTEGER,
    live_size REAL,
    backtest_size REAL,
    size_delta_pct REAL
);

-- P1 stateful min-hold: per-coin position state carried across daily cycles so
-- the live runner reproduces v2_sizing.build_positions_with_hold (7-day min
-- hold + adaptive early exit). entry_base is the PRE-trend signed sleeve frozen
-- at entry/flip; the runner re-applies the current bar's SMA multiplier.
CREATE TABLE IF NOT EXISTS hold_state (
    coin TEXT PRIMARY KEY,
    current_dir INTEGER NOT NULL,
    bars_held INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_base REAL NOT NULL,
    entry_cycle TEXT,
    updated_ts TEXT
);

-- Hybrid modulator outputs (one row per coin per hybrid cycle). Additive;
-- quant journals simply never write it. fallback=1 means the modulator
-- failed/was skipped and the hybrid traded pure quant (1.0, 0.0).
CREATE TABLE IF NOT EXISTS modulator_outputs (
    cycle_id TEXT NOT NULL,
    coin TEXT NOT NULL,
    multiplier REAL NOT NULL,
    effective_weight REAL NOT NULL,
    llm_confidence REAL,
    regime TEXT,
    fallback INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (cycle_id, coin)
);
