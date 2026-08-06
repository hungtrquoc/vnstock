"""
screener_store.py
-------------------
Luu tru KET QUA quet toan thi truong (Tab "Quet toan thi truong") CHO BAN
WEB - dung Postgres, qua NHIEU LAN goi /api/screener/cron (moi lan chi quet
1 lo (batch) nho, vi Vercel serverless function co gioi han thoi gian chay -
xem README.md ve ly do va cach kich hoat cron nhieu lan/ngay).

QUAN TRONG: module nay CHI dung cho ban web, KHONG dung cho app Windows (ban
desktop quet truc tiep trong RAM qua ScreenerWorker trong main.py, hien ket
qua ngay tren UI, khong can luu tien do giua cac lan quet) - vi vay module
nay nam TRONG api/ (khong o thu muc goc cung main.py/analysis_engine.py...),
de ro rang day la phan RIENG cua ban web, main.py khong bao gio import module
nay.

2 bang (xem schema.sql de xem dinh nghia day du):
  - screener_results: ket qua MOI NHAT cho tung ma (UPSERT, khong luu lich
    su cac lan quet cu - giong nhu screen_table trong main.py chi hien trang
    thai hien tai).
  - screener_progress: 1 dong duy nhat (singleton, id=1) luu danh sach ma
    dang quet trong "vong" hien tai + vi tri (cursor) da quet toi dau - de
    lan goi cron TIEP THEO biet quet tiep tu dau, khong quet lai tu dau mỗi
    lan (se khong bao gio quet het ~1700 ma neu lam vay).

Dung placeholder '%s' (cu phap psycopg2) TRUC TIEP, KHONG qua ham dich
data_manager._sql() - vi module nay chi chay tren Postgres (ban web), khac
data_manager.py la module dung CHUNG ca 2 backend.
"""
from __future__ import annotations

import json
from typing import Optional

import market_screener as screener


def ensure_tables(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS screener_results (
            symbol             TEXT PRIMARY KEY,
            status             TEXT NOT NULL,
            message            TEXT,
            classic_verdict    TEXT,
            classic_positive   BOOLEAN,
            stat_positive      BOOLEAN,
            stat_best_horizon  INTEGER,
            stat_best_hit_rate REAL,
            stat_best_pvalue   REAL,
            scanned_at         TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS screener_progress (
            id                  INTEGER PRIMARY KEY DEFAULT 1,
            symbols_json        TEXT,
            cursor_pos          INTEGER DEFAULT 0,
            round_started_at    TIMESTAMPTZ,
            round_completed_at  TIMESTAMPTZ,
            updated_at          TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    conn.commit()


def get_progress(conn) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT symbols_json, cursor_pos, round_started_at, round_completed_at "
        "FROM screener_progress WHERE id = 1"
    )
    row = cur.fetchone()
    if row is None:
        return None
    symbols_json, cursor_pos, started, completed = row
    return {
        "symbols": json.loads(symbols_json) if symbols_json else [],
        "cursor_pos": cursor_pos or 0,
        "round_started_at": started,
        "round_completed_at": completed,
    }


def start_new_round(conn, symbols: list[str]) -> None:
    """Bat dau 1 'vong' quet moi: nap lai toan bo danh sach ma, dua cursor
    ve 0. Goi khi chua co progress nao, HOAC vong truoc da quet het (cursor
    >= len(symbols)) - lien tuc lap lai, coi nhu 'quet lien tuc toan thi
    truong', khong bao gio dung han."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO screener_progress (id, symbols_json, cursor_pos, round_started_at, round_completed_at, updated_at)
        VALUES (1, %s, 0, now(), NULL, now())
        ON CONFLICT (id) DO UPDATE SET
            symbols_json = EXCLUDED.symbols_json, cursor_pos = 0,
            round_started_at = now(), round_completed_at = NULL, updated_at = now()
        """,
        (json.dumps(symbols),),
    )
    conn.commit()


def advance_cursor(conn, new_pos: int, round_done: bool) -> None:
    cur = conn.cursor()
    if round_done:
        cur.execute(
            "UPDATE screener_progress SET cursor_pos = %s, round_completed_at = now(), updated_at = now() WHERE id = 1",
            (new_pos,),
        )
    else:
        cur.execute(
            "UPDATE screener_progress SET cursor_pos = %s, updated_at = now() WHERE id = 1",
            (new_pos,),
        )
    conn.commit()


def save_result(conn, row: "screener.ScreenerRow") -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO screener_results
            (symbol, status, message, classic_verdict, classic_positive,
             stat_positive, stat_best_horizon, stat_best_hit_rate, stat_best_pvalue, scanned_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (symbol) DO UPDATE SET
            status = EXCLUDED.status, message = EXCLUDED.message,
            classic_verdict = EXCLUDED.classic_verdict, classic_positive = EXCLUDED.classic_positive,
            stat_positive = EXCLUDED.stat_positive, stat_best_horizon = EXCLUDED.stat_best_horizon,
            stat_best_hit_rate = EXCLUDED.stat_best_hit_rate, stat_best_pvalue = EXCLUDED.stat_best_pvalue,
            scanned_at = now()
        """,
        (
            row.symbol, row.status, row.message, row.classic_verdict, row.classic_positive,
            row.stat_positive, row.stat_best_horizon, row.stat_best_hit_rate, row.stat_best_pvalue,
        ),
    )
    conn.commit()


def get_results(conn, verdict_filter: Optional[list[str]] = None, stat_only: bool = False) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT symbol, status, message, classic_verdict, classic_positive, stat_positive, "
        "stat_best_horizon, stat_best_hit_rate, stat_best_pvalue, scanned_at FROM screener_results "
        "ORDER BY symbol ASC"
    )
    cols = ["symbol", "status", "message", "classic_verdict", "classic_positive", "stat_positive",
            "stat_best_horizon", "stat_best_hit_rate", "stat_best_pvalue", "scanned_at"]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    if stat_only:
        rows = [r for r in rows if r["stat_positive"]]
    if verdict_filter is not None:
        rows = [
            r for r in rows
            if (r["classic_verdict"] or "") in verdict_filter
        ]
    return rows
