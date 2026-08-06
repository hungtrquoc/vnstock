"""
api/index.py
-------------
API web (FastAPI) cho VN Stock Analysis App. DUNG CHUNG 1 THU MUC voi ban
desktop Windows (theo yeu cau nguoi dung 2026-08-06: "dùng chung 1 thư mục
cho cả web/app") - analysis_engine.py, classic_ta.py, candlestick_patterns.py,
sector_map.py, market_screener.py, chart_view.py, data_manager.py o day
CHINH LA cac file dang dung cho ban desktop (main.py), khong phai ban copy
rieng - sua 1 lan la ca 2 ban cung nhan duoc thay doi ngay (khong con nguy co
lech nhau giua 2 thu muc nhu thiet ke ban dau).

Deploy tren Vercel (Python runtime cho FastAPI, xem
https://vercel.com/docs/functions/runtimes/python) - Vercel tim bien `app`
(FastAPI instance) trong file nay va tu dong bien thanh 1 serverless
function ASGI. vercel.json rewrite TOAN BO /api/* ve dung file nay (1 function
duy nhat xu ly moi route qua FastAPI router, thay vi 1 file .py rieng cho
tung endpoint) - do la cach pho bien nhat cho FastAPI tren Vercel.

QUAN TRONG (rui ro chua kiem chung - doc README.md): requirements.txt o goc
repo (VN_StockApp/requirements.txt) la cho ban DESKTOP, co PyQt6 - KHONG can
va co the lam hong build Python tren Vercel (PyQt6 can thu vien he thong
Qt, khong cai duoc/khong can tren serverless). File `api/requirements.txt`
(cung cap voi file nay) CHI list dung cac thu vien can cho web (fastapi/
pandas/numpy/requests/matplotlib/psycopg2-binary, KHONG co PyQt6). Theo tai
lieu Vercel, requirements.txt nam CUNG cap voi ham (api/) duoc uu tien hon
requirements.txt o goc - nhung DIEU NAY CHUA duoc kiem chung thuc te (khong
co tai khoan Vercel de deploy thu trong qua trinh phat trien). Neu build tren
Vercel bao loi lien quan PyQt6, day chinh la nguyen nhan - xem README.md.

Bien moi truong CAN THIET (dat trong Vercel Project Settings -> Environment
Variables, xem README.md):
  - DATABASE_URL: connection string Postgres (Supabase/Neon...), dang
    "postgres://user:pass@host:port/dbname".
  - CRON_SECRET: chuoi bi mat tuy chon, bao ve endpoint /api/screener/cron
    khoi bi nguoi la goi tran (xem _require_cron_secret).
"""
from __future__ import annotations

import base64
import io
import os
import sys
from datetime import datetime, timezone

# Them thu muc GOC cua repo (1 cap tren api/) vao sys.path, de co the
# `import analysis_engine`, `import data_manager`... - cac file nay nam
# ngay tai goc repo (dung chung voi main.py cua ban desktop), khong con
# trong 1 thu muc backend/ rieng nhu thiet ke ban dau nua.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))
# Them CA thu muc api/ chinh nginh (ban than __file__) vao sys.path -
# BAT BUOC tren Vercel: runtime tai file nay bang importlib.exec_module()
# (khong phai chay truc tiep "python api/index.py"), nen Python KHONG tu
# dong them thu muc chua file dang chay vao sys.path nhu binh thuong. Thieu
# dong nay se bi "ModuleNotFoundError: No module named 'screener_store'"
# du file api/screener_store.py van nam dung cho, van duoc dong goi day du
# (da xac nhan qua log Vercel thuc te 2026-08-06) - khong phai loi thieu file.
sys.path.insert(0, _THIS_DIR)

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # bat buoc tren serverless - khong co display server
import matplotlib.pyplot as plt  # noqa: E402

import data_manager as dm  # noqa: E402
import analysis_engine as ae  # noqa: E402
import classic_ta  # noqa: E402
import candlestick_patterns as cp  # noqa: E402
import chart_view as cv  # noqa: E402
import market_screener as screener  # noqa: E402
import sector_map  # noqa: E402
import screener_store as store  # noqa: E402 - cung thu muc api/, chi dung cho web

app = FastAPI(title="VN Stock Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # trang tinh va API cung 1 domain Vercel - de mo cho don gian, co the sua lai sau
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.environ.get("DATABASE_URL")
CRON_SECRET = os.environ.get("CRON_SECRET")


def get_conn():
    if not DATABASE_URL:
        raise HTTPException(
            500,
            "Chưa cấu hình DATABASE_URL (env var) trên Vercel - xem README.md phần "
            "'Thiết lập Postgres (Supabase/Neon)'.",
        )
    return dm.get_connection(DATABASE_URL)


# ---------------------------------------------------------------------------
# Tab 1 (tuong duong): xem/cap nhat du lieu 1 ma
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat(), "db_configured": bool(DATABASE_URL)}


@app.post("/api/symbols/{symbol}/update")
def update_symbol(symbol: str, full: bool = False):
    conn = get_conn()
    try:
        result = dm.full_refresh(conn, symbol) if full else dm.update_stock_data(conn, symbol)
    finally:
        conn.close()
    return {
        "symbol": result.symbol,
        "action": result.action,
        "message": result.message,
        "rows_before": result.rows_before,
        "rows_after": result.rows_after,
    }


@app.get("/api/symbols/{symbol}/preview")
def preview_symbol(symbol: str, n: int = 20):
    conn = get_conn()
    try:
        df = dm.get_price_df(conn, symbol.strip().upper())
    finally:
        conn.close()
    if df.empty:
        return {"symbol": symbol.upper(), "rows": []}
    tail = df.tail(n)
    rows = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "open": None if r["open"] != r["open"] else float(r["open"]),
            "high": None if r["high"] != r["high"] else float(r["high"]),
            "low": None if r["low"] != r["low"] else float(r["low"]),
            "close": None if r["close"] != r["close"] else float(r["close"]),
            "volume": None if r["volume"] != r["volume"] else int(r["volume"]),
        }
        for d, r in tail.iterrows()
    ]
    return {"symbol": symbol.upper(), "rows": rows}


# ---------------------------------------------------------------------------
# Tab 2 (tuong duong): phan tich 1 ma - dung LAI y het logic
# main._run_full_analysis / _render_analysis_result trong ban desktop, chi
# doi output tu widget PyQt6 sang JSON (+ chart PNG dang base64).
# ---------------------------------------------------------------------------

@app.get("/api/analyze/{symbol}")
def analyze(symbol: str):
    symbol = symbol.strip().upper()
    conn = get_conn()
    try:
        fetch_warnings: list[str] = []
        vn_result = dm.update_stock_data(conn, "VNINDEX")
        if vn_result.action == "error":
            fetch_warnings.append(f"Không cập nhật được VNINDEX mới nhất ({vn_result.message}).")
        sym_result = dm.update_stock_data(conn, symbol)
        if sym_result.action == "error":
            fetch_warnings.append(f"Không cập nhật được {symbol} mới nhất ({sym_result.message}).")

        stock_df = dm.get_price_df(conn, symbol)
        vnindex_df = dm.get_price_df(conn, "VNINDEX")
        if stock_df.empty:
            raise HTTPException(400, f"Không có/không tải được dữ liệu cho {symbol}.")
        if vnindex_df.empty:
            raise HTTPException(400, "Không có/không tải được dữ liệu VNINDEX.")
        if len(stock_df) < ae.MIN_USABLE_ROWS + ae.WARMUP_ROWS:
            raise HTTPException(
                400,
                f"{symbol} chỉ có {len(stock_df)} dòng dữ liệu - cần ít nhất "
                f"~{ae.MIN_USABLE_ROWS + ae.WARMUP_ROWS} dòng để phân tích có ý nghĩa.",
            )

        peer_price_dfs = {}
        for peer_symbol in sector_map.get_peers(symbol):
            try:
                dm.update_stock_data(conn, peer_symbol)
                peer_df = dm.get_price_df(conn, peer_symbol)
                if not peer_df.empty:
                    peer_price_dfs[peer_symbol] = peer_df
            except Exception:
                continue

        assessment = ae.generate_assessment(symbol, stock_df, vnindex_df, peer_price_dfs=peer_price_dfs)
        if fetch_warnings:
            assessment.warnings = fetch_warnings + assessment.warnings

        ind = ae.compute_indicators(stock_df)
        candle_hits = cp.recent_patterns(stock_df, n_recent=15)
        scorecard = classic_ta.build_classic_scorecard(ind.iloc[-1], candle_hits)

        metrics = []
        for h in assessment.forecasts.keys():
            f = assessment.forecasts[h]
            m = assessment.oot_metrics.get(h)
            b = assessment.baseline_oot_metrics.get(h)
            p = assessment.p_values.get(h)
            metrics.append(
                {
                    "horizon": h,
                    "expected_pct": f["expected_return_pct"],
                    "expected_price": f["price_path"][-1],
                    "lower_pct": f["lower_pct"],
                    "lower_price": f["lower_price_path"][-1],
                    "upper_pct": f["upper_pct"],
                    "upper_price": f["upper_price_path"][-1],
                    "hit_rate": m.hit_rate if (m and m.n_samples > 0) else None,
                    "baseline_hit_rate": b.hit_rate if (b and b.n_samples > 0) else None,
                    "p_value": p,
                    "significant": bool(p is not None and p < ae.SIGNIFICANCE_ALPHA),
                    "n_samples": m.n_samples if m else 0,
                }
            )

        chart_png_base64 = None
        try:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            cv.plot_analysis(ax, symbol, stock_df, assessment)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            chart_png_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            chart_png_base64 = None  # bieu do chi la "them", loi ve chart khong duoc lam hong ca ket qua phan tich

        return {
            "symbol": symbol,
            "as_of_date": assessment.as_of_date,
            "current_price": assessment.current_price,
            "warnings": assessment.warnings,
            "metrics": metrics,
            "chart_png_base64": chart_png_base64,
            "classic_verdict": scorecard.verdict,
            "classic_verdict_detail": scorecard.verdict_detail,
            "classic_signals": [
                {"name": s.name, "direction": s.direction, "detail": s.detail} for s in scorecard.signals
            ],
            "classic_disclaimer": scorecard.disclaimer,
            "peers_used": assessment.peers_used,
            "calibrated_k": assessment.calibrated_k,
            "n_folds_used": assessment.n_folds_used,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tab 3 (tuong duong): quet toan thi truong - THIET KE KHAC ban desktop, vi
# Vercel serverless function co gioi han thoi gian chay (khong the quet
# ~1700 ma trong 1 request nhu ScreenerWorker cua main.py). Thay vao do:
# /api/screener/cron chi quet 1 LO NHO moi lan goi, luu tien do vao Postgres
# (screener_store.py); /api/screener/results DOC ket qua da luu san - web
# hien "ket qua cua lan quet gan nhat", khong phai "bam nut la quet xong
# ngay" nhu ban desktop. Xem README.md ve cach kich hoat /api/screener/cron
# lap lai nhieu lan/ngay (Vercel Cron tren goi Hobby chi cho 1 lan/ngay).
# ---------------------------------------------------------------------------

def _require_cron_secret(request: Request) -> None:
    if not CRON_SECRET:
        return  # chua cau hinh secret -> khong bat buoc (tien loi khi moi setup, nen dat secret khi public)
    auth = request.headers.get("authorization", "")
    provided = request.query_params.get("secret", "")
    if auth != f"Bearer {CRON_SECRET}" and provided != CRON_SECRET:
        raise HTTPException(401, "Thiếu hoặc sai CRON_SECRET.")


@app.get("/api/screener/cron")
def screener_cron(request: Request, batch_size: int = 30):
    _require_cron_secret(request)
    conn = get_conn()
    try:
        store.ensure_tables(conn)
        progress = store.get_progress(conn)
        if progress is None or not progress["symbols"] or progress["cursor_pos"] >= len(progress["symbols"]):
            symbols = screener.fetch_all_listed_symbols()
            if not symbols:
                symbols = screener.FALLBACK_SYMBOLS
            store.start_new_round(conn, symbols)
            progress = store.get_progress(conn)

        symbols = progress["symbols"]
        start = progress["cursor_pos"]
        batch = symbols[start : start + batch_size]
        scanned = []
        for sym in batch:
            try:
                row = screener.screen_one_symbol(conn, sym)
            except Exception as e:  # noqa: BLE001
                row = screener.ScreenerRow(sym, "error", message=str(e))
            store.save_result(conn, row)
            scanned.append(sym)

        new_pos = start + len(batch)
        round_done = new_pos >= len(symbols)
        store.advance_cursor(conn, new_pos, round_done)

        return {
            "scanned_this_batch": scanned,
            "cursor_pos": new_pos,
            "total_symbols": len(symbols),
            "round_done": round_done,
        }
    finally:
        conn.close()


@app.get("/api/screener/results")
def screener_results(
    positive: bool = True,
    neutral: bool = True,
    negative: bool = True,
    unrated: bool = False,
    stat_only: bool = False,
):
    verdicts = []
    if positive:
        verdicts.append("Tích cực")
    if neutral:
        verdicts.extend(["Trung lập", "Trung lập/hỗn hợp"])
    if negative:
        verdicts.append("Tiêu cực")
    if unrated:
        verdicts.append("")

    conn = get_conn()
    try:
        store.ensure_tables(conn)
        rows = store.get_results(conn, verdict_filter=verdicts, stat_only=stat_only)
        progress = store.get_progress(conn)
    finally:
        conn.close()

    return {
        "rows": rows,
        "progress": {
            "cursor_pos": progress["cursor_pos"] if progress else 0,
            "total_symbols": len(progress["symbols"]) if progress else 0,
            "round_started_at": str(progress["round_started_at"]) if progress and progress["round_started_at"] else None,
            "round_completed_at": str(progress["round_completed_at"]) if progress and progress["round_completed_at"] else None,
        },
    }
