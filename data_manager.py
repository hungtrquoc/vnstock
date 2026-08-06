"""
data_manager.py
----------------
Lop du lieu cho VN Stock Analysis App.

Trach nhiem duy nhat cua module nay: quan ly du lieu gia lich su co phieu
(tai ve tu VNDIRECT, luu vao DB, cap nhat/refresh). KHONG chua logic
phan tich ky thuat - phan tich nam o module rieng (analysis_engine.py, Phase 2).

Logic theo yeu cau (Buoc 1-2):
- Nhap ma co phieu -> kiem tra DB da co du lieu chua.
- Chua co: tai toan bo lich su tu 2000-01-01 den hien tai, luu vao DB.
- Da co: xoa dong du lieu GAN NHAT (vi co the la du lieu intraday chua phai EOD),
  roi tai lai du lieu tu ngay do den hien tai, upsert vao DB.
- Tinh nang rieng: "Keo lai toan bo du lieu" - xoa het du lieu cu cua ma do va
  tai lai tu dau (dung khi co phieu vua chia co tuc/tach co phieu lam lech gia qua khu).

HO TRO 2 BACKEND DB (them 2026-08-06, theo yeu cau nguoi dung: "muon co them
phan UI de chay tren vercel... phan backend dung chung nhe"):
  - SQLite (file local) - dung cho app Windows (PyQt6), HANH VI GIU NGUYEN 100%
    nhu truoc, khong doi gi cho nguoi dung dang dung ban desktop.
  - Postgres (Supabase/Neon...) - dung cho ban web tren Vercel (nhieu request
    dong thoi tu serverless function, can 1 DB host tren internet - SQLite file
    khong phu hop cho truong hop nay).
  Day la MODULE DUY NHAT can biet ve loai DB dang dung. Moi module khac
  (analysis_engine.py, classic_ta.py, candlestick_patterns.py, sector_map.py,
  market_screener.py, va ca 2 UI - main.py PyQt6 va API web) deu CHI goi cac
  ham public o day (get_connection/get_price_df/upsert_prices/
  update_stock_data/full_refresh/...) - khong ai tu viet SQL truc tiep - nen
  chung hoat dong GIONG NHAU tren ca 2 backend ma khong can sua 1 dong nao ca.
  Chon backend hoan toan tu dong trong get_connection(): neu chuoi truyen vao
  bat dau bang "postgres://" hoac "postgresql://" -> dung psycopg2 (Postgres);
  nguoc lai coi la duong dan file -> dung sqlite3 (hanh vi CU). psycopg2 chi
  duoc import LUOI (trong nhanh Postgres) nen app Windows KHONG can cai
  psycopg2 - giu nguyen dependency footprint cua ban desktop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

VNDIRECT_URL = "https://dchart-api.vndirect.com.vn/dchart/history"
USER_AGENT = "Mozilla/5.0 (compatible; VNStockApp/1.0)"
DEFAULT_FROM_DATE = "2000-01-01"


@dataclass
class UpdateResult:
    symbol: str
    action: str          # "full_pull" | "incremental_update" | "full_refresh" | "no_change" | "error"
    rows_before: int
    rows_after: int
    rows_added: int
    last_date_before: Optional[str]
    last_date_after: Optional[str]
    message: str = ""


# ---------------------------------------------------------------------------
# DB layer - dual backend (SQLite / Postgres), xem docstring module o tren.
# ---------------------------------------------------------------------------

def _is_pg(conn) -> bool:
    """Nhan biet connection dang la psycopg2 (Postgres) hay sqlite3 - dung de
    quyet dinh co can dich placeholder '?' -> '%s' hay khong. Khong import
    psycopg2 o day (chi kiem tra ten module cua object, khong cap phat gi
    moi) de nhanh nay khong ep app Windows phai co psycopg2 cai san."""
    return conn.__class__.__module__.startswith("psycopg2")


def _sql(conn, query: str) -> str:
    """Dich cu phap placeholder tham so: sqlite3 dung '?', psycopg2 dung
    '%s'. Cho phep TOAN BO ham duoi day viet SQL 1 LAN DUY NHAT (theo cu
    phap sqlite quen thuoc trong code cu cua project), dung chung cho ca 2
    backend - khong phai duy tri 2 bo cau SQL song song."""
    return query.replace("?", "%s") if _is_pg(conn) else query


def get_connection(db_path_or_url: str | Path):
    """Mo (va tao moi neu chua co) connection DB. Nhan 1 trong 2 dang:
      - duong dan file (vd Path.home()/"VNStockApp"/"vn_stock_data.db")
        -> SQLite, dung cho app Windows, HANH VI GIU NGUYEN nhu truoc.
      - chuoi bat dau bang "postgres://" hoac "postgresql://"
        -> Postgres (vd DATABASE_URL cua Supabase/Neon), dung cho ban web.
    Ca 2 nhanh deu goi init_db() de dam bao schema ton tai truoc khi tra ve."""
    dsn = str(db_path_or_url)
    if dsn.startswith("postgres://") or dsn.startswith("postgresql://"):
        import psycopg2  # import luoi - chi can khi thuc su dung Postgres
        conn = psycopg2.connect(dsn)
        init_db(conn)
        return conn

    import sqlite3
    db_path = Path(dsn)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: app nay dung 1 connection duy nhat nhung goi
    # tu ca main thread (UI) va QThread (WorkerThread chay fetch/update). UI
    # da tu khoa nut bam trong luc worker chay (_set_busy) nen khong bao gio
    # co 2 thread ghi DB dong thoi - an toan de tat kiem tra same-thread cua
    # sqlite3. Neu khong tat, se gap loi "SQLite objects created in a thread
    # can only be used in that same thread."
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    init_db(conn)
    return conn


def init_db(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol TEXT NOT NULL,
            date   TEXT NOT NULL,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume BIGINT,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS symbols_meta (
            symbol       TEXT PRIMARY KEY,
            first_date   TEXT,
            last_date    TEXT,
            last_synced  TEXT,
            row_count    INTEGER
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol ON daily_prices(symbol)")
    conn.commit()


def has_data(conn, symbol: str) -> bool:
    cur = conn.cursor()
    cur.execute(_sql(conn, "SELECT 1 FROM daily_prices WHERE symbol = ? LIMIT 1"), (symbol,))
    return cur.fetchone() is not None


def get_last_date(conn, symbol: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute(_sql(conn, "SELECT MAX(date) FROM daily_prices WHERE symbol = ?"), (symbol,))
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def get_row_count(conn, symbol: str) -> int:
    cur = conn.cursor()
    cur.execute(_sql(conn, "SELECT COUNT(*) FROM daily_prices WHERE symbol = ?"), (symbol,))
    return cur.fetchone()[0]


def get_price_df(conn, symbol: str) -> pd.DataFrame:
    """Doc toan bo du lieu cua 1 ma tu DB, tra ve DataFrame index la date (datetime)."""
    query = _sql(
        conn,
        "SELECT date, open, high, low, close, volume FROM daily_prices "
        "WHERE symbol = ? ORDER BY date ASC",
    )
    df = pd.read_sql_query(query, conn, params=(symbol,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def upsert_prices(conn, symbol: str, df: pd.DataFrame) -> int:
    """Insert-or-replace cac dong gia vao DB. df phai co index la date va cac
    cot open/high/low/close/volume. Tra ve so dong da upsert."""
    if df is None or df.empty:
        return 0
    rows = [
        (
            symbol,
            d.strftime("%Y-%m-%d"),
            float(r["open"]) if pd.notna(r["open"]) else None,
            float(r["high"]) if pd.notna(r["high"]) else None,
            float(r["low"]) if pd.notna(r["low"]) else None,
            float(r["close"]) if pd.notna(r["close"]) else None,
            int(r["volume"]) if pd.notna(r["volume"]) else None,
        )
        for d, r in df.iterrows()
    ]
    cur = conn.cursor()
    cur.executemany(
        _sql(
            conn,
            """
            INSERT INTO daily_prices (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, date) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume
            """,
        ),
        rows,
    )
    conn.commit()
    _refresh_meta(conn, symbol)
    return len(rows)


def delete_symbol_data(conn, symbol: str) -> int:
    cur = conn.cursor()
    cur.execute(_sql(conn, "DELETE FROM daily_prices WHERE symbol = ?"), (symbol,))
    deleted = cur.rowcount
    conn.commit()
    cur.execute(_sql(conn, "DELETE FROM symbols_meta WHERE symbol = ?"), (symbol,))
    conn.commit()
    return deleted


def delete_last_row(conn, symbol: str) -> Optional[str]:
    """Xoa dong co ngay lon nhat cua 1 ma. Tra ve ngay da xoa (hoac None neu khong co)."""
    last_date = get_last_date(conn, symbol)
    if last_date is None:
        return None
    cur = conn.cursor()
    cur.execute(_sql(conn, "DELETE FROM daily_prices WHERE symbol = ? AND date = ?"), (symbol, last_date))
    conn.commit()
    _refresh_meta(conn, symbol)
    return last_date


def _refresh_meta(conn, symbol: str) -> None:
    cur = conn.cursor()
    cur.execute(
        _sql(conn, "SELECT MIN(date), MAX(date), COUNT(*) FROM daily_prices WHERE symbol = ?"),
        (symbol,),
    )
    first_d, last_d, cnt = cur.fetchone()
    now_iso = datetime.now(timezone.utc).isoformat()
    cur.execute(
        _sql(
            conn,
            """
            INSERT INTO symbols_meta (symbol, first_date, last_date, last_synced, row_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                first_date=excluded.first_date, last_date=excluded.last_date,
                last_synced=excluded.last_synced, row_count=excluded.row_count
            """,
        ),
        (symbol, first_d, last_d, now_iso, cnt),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# VNDIRECT fetch layer
# (Da verify hoat dong trong cac notebook cu cua nguoi dung - giu nguyen
#  format request/response, chi bo sung retry + backoff + loc chat luong.)
# Hoan toan khong phu thuoc DB - dung CHUNG y het nhau cho ca app Windows va
# ban web (Vercel), khong can sua gi.
# ---------------------------------------------------------------------------

def fetch_vndirect_data(
    symbol: str,
    from_date_str: str = DEFAULT_FROM_DATE,
    to_date_str: Optional[str] = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Tai du lieu OHLCV theo ngay tu VNDIRECT dchart API.

    Tra ve DataFrame voi index la Timestamp (ngay, khong gio) va cac cot
    open/high/low/close/volume. Tra ve DataFrame rong neu loi hoac khong co du lieu.
    """
    from_dt = datetime.strptime(from_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    from_ts = int(from_dt.timestamp())

    if to_date_str:
        to_dt = datetime.strptime(to_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        to_ts = int(to_dt.timestamp()) + 86400  # bao gom het ngay to_date
    else:
        # VNDIRECT server o gio VN (UTC+7); cong them 7h de chac chan lay du phien hom nay
        to_ts = int((datetime.now(timezone.utc) + timedelta(hours=7)).timestamp())

    params = {"symbol": symbol, "resolution": "D", "from": from_ts, "to": to_ts}
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(max_retries):
        try:
            resp = requests.get(VNDIRECT_URL, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("s") != "ok" or not data.get("t"):
                return pd.DataFrame()

            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") for ts in data["t"]]
                    ),
                    "open": data["o"],
                    "high": data["h"],
                    "low": data["l"],
                    "close": data["c"],
                    "volume": data["v"],
                }
            ).set_index("date").sort_index()

            # loc du lieu bat thuong (giong logic da verify trong notebook cu)
            df = df[
                (df["volume"] >= 0)
                & (df["close"] > 0)
                & (df["high"] >= df["close"])
                & (df["low"] <= df["close"])
            ]
            df["close"] = df["close"].replace(0, np.nan).ffill()
            return df
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    # het retry, tra ve rong
    return pd.DataFrame()


def fetch_vnindex(from_date_str: str = DEFAULT_FROM_DATE, to_date_str: Optional[str] = None) -> pd.DataFrame:
    return fetch_vndirect_data("VNINDEX", from_date_str, to_date_str)


# ---------------------------------------------------------------------------
# Update orchestration (Buoc 2 cua flow)
# ---------------------------------------------------------------------------

def update_stock_data(conn, symbol: str) -> UpdateResult:
    """
    - Chua co du lieu -> tai toan bo lich su.
    - Da co du lieu -> xoa dong gan nhat, tai lai tu ngay do den hien tai, upsert.
    """
    symbol = symbol.strip().upper()
    rows_before = get_row_count(conn, symbol)
    last_date_before = get_last_date(conn, symbol)

    if rows_before == 0:
        df = fetch_vndirect_data(symbol, DEFAULT_FROM_DATE)
        if df.empty:
            return UpdateResult(
                symbol, "error", 0, 0, 0, None, None,
                message=f"Không tải được dữ liệu cho {symbol} (kiểm tra lại mã cổ phiếu hoặc kết nối mạng).",
            )
        added = upsert_prices(conn, symbol, df)
        rows_after = get_row_count(conn, symbol)
        return UpdateResult(
            symbol, "full_pull", 0, rows_after, added, None, get_last_date(conn, symbol),
            message=f"Đã tải mới {added} dòng dữ liệu cho {symbol}.",
        )

    # da co du lieu -> xoa dong gan nhat roi keo lai tu ngay do
    refetch_from = delete_last_row(conn, symbol)  # tra ve ngay vua xoa
    if refetch_from is None:
        refetch_from = DEFAULT_FROM_DATE

    df = fetch_vndirect_data(symbol, refetch_from)
    if df.empty:
        rows_after = get_row_count(conn, symbol)
        return UpdateResult(
            symbol, "error", rows_before, rows_after, 0, last_date_before, get_last_date(conn, symbol),
            message=f"Đã xóa dòng {refetch_from} nhưng không tải lại được dữ liệu mới. "
                    f"DB hiện có {rows_after} dòng (thiếu 1 dòng so với trước). Hãy thử lại.",
        )
    added = upsert_prices(conn, symbol, df)
    rows_after = get_row_count(conn, symbol)
    return UpdateResult(
        symbol, "incremental_update", rows_before, rows_after, added,
        last_date_before, get_last_date(conn, symbol),
        message=f"Đã cập nhật {symbol}: xóa lại từ {refetch_from}, upsert {added} dòng. "
                f"Tổng số dòng: {rows_after}.",
    )


def full_refresh(conn, symbol: str) -> UpdateResult:
    """Keo lai TOAN BO du lieu lich su cho 1 ma (dung khi co chia co tuc/tach co phieu)."""
    symbol = symbol.strip().upper()
    rows_before = get_row_count(conn, symbol)
    last_date_before = get_last_date(conn, symbol)

    delete_symbol_data(conn, symbol)
    df = fetch_vndirect_data(symbol, DEFAULT_FROM_DATE)
    if df.empty:
        return UpdateResult(
            symbol, "error", rows_before, 0, 0, last_date_before, None,
            message=f"Đã xóa dữ liệu cũ của {symbol} nhưng không tải lại được dữ liệu mới. "
                    f"DB cho mã này hiện đang RỖNG - hãy thử lại ngay.",
        )
    added = upsert_prices(conn, symbol, df)
    rows_after = get_row_count(conn, symbol)
    return UpdateResult(
        symbol, "full_refresh", rows_before, rows_after, added, last_date_before, get_last_date(conn, symbol),
        message=f"Đã kéo lại toàn bộ dữ liệu cho {symbol}: {rows_after} dòng (từ {get_last_date(conn, symbol)}).",
    )
