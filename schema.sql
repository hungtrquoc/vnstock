-- schema.sql
-- -----------
-- Schema Postgres cho ban web (Supabase/Neon...). KHONG BAT BUOC phai chay
-- tay script nay - data_manager.init_db() va api/screener_store.ensure_tables()
-- da tu dong CREATE TABLE IF NOT EXISTS moi lan API khoi dong, nen chi can
-- dat DATABASE_URL dung la du. Script nay chi de THAM KHAO/kiem tra truoc
-- trong SQL editor cua Supabase/Neon neu muon xem truoc schema se nhu the
-- nao, hoac de chay tay 1 lan cho chac chan truoc khi deploy.

-- 2 bang nay giong 100% schema SQLite cua ban desktop (Windows) - dinh
-- nghia trong data_manager.py, dung CHUNG cho ca 2 backend.
CREATE TABLE IF NOT EXISTS daily_prices (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,      -- 'YYYY-MM-DD'
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume BIGINT,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol ON daily_prices(symbol);

CREATE TABLE IF NOT EXISTS symbols_meta (
    symbol       TEXT PRIMARY KEY,
    first_date   TEXT,
    last_date    TEXT,
    last_synced  TEXT,   -- ISO timestamp cua lan cap nhat gan nhat
    row_count    INTEGER
);

-- 2 bang duoi day CHI dung cho ban web (api/screener_store.py) - de luu
-- tien do quet toan thi truong qua nhieu lan goi /api/screener/cron (xem
-- README.md va docstring cua api/screener_store.py de hieu ro thiet ke).
CREATE TABLE IF NOT EXISTS screener_results (
    symbol             TEXT PRIMARY KEY,
    status             TEXT NOT NULL,       -- 'ok' | 'skip' | 'error'
    message            TEXT,
    classic_verdict    TEXT,                -- 'Tích cực' / 'Trung lập...' / 'Tiêu cực' / NULL
    classic_positive   BOOLEAN,
    stat_positive      BOOLEAN,
    stat_best_horizon  INTEGER,
    stat_best_hit_rate REAL,
    stat_best_pvalue   REAL,
    scanned_at         TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS screener_progress (
    id                  INTEGER PRIMARY KEY DEFAULT 1,   -- luon = 1 (singleton, chi 1 dong)
    symbols_json        TEXT,        -- JSON list toan bo ma dang quet trong vong hien tai
    cursor_pos          INTEGER DEFAULT 0,   -- da quet toi vi tri nao trong danh sach tren
    round_started_at    TIMESTAMPTZ,
    round_completed_at  TIMESTAMPTZ,          -- NULL khi vong hien tai chua quet xong
    updated_at          TIMESTAMPTZ DEFAULT now()
);
