"""
Test data_manager.py logic bang mock fetch (khong goi network thuc, vi sandbox
build bi chan domain VNDIRECT). Nguoi dung nen chay lai test nay + app thuc te
tren may minh de xac nhan fetch mang hoat dong dung.
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import data_manager as dm


def make_fake_df(start, end):
    idx = pd.bdate_range(start, end)  # ngay lam viec, gia lap phien giao dich
    n = len(idx)
    base = 50.0
    data = {
        "open": [base + i * 0.1 for i in range(n)],
        "high": [base + i * 0.1 + 0.5 for i in range(n)],
        "low": [base + i * 0.1 - 0.5 for i in range(n)],
        "close": [base + i * 0.1 + 0.2 for i in range(n)],
        "volume": [1000 + i for i in range(n)],
    }
    return pd.DataFrame(data, index=idx)


def run_tests():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    conn = dm.get_connection(db_path)

    symbol = "FAKE"
    today = datetime(2026, 7, 29)

    # --- Test 1: fresh symbol -> full pull ---
    full_range_end = today - timedelta(days=1)
    fake_full = make_fake_df("2026-01-01", full_range_end.strftime("%Y-%m-%d"))

    def fetch_stub_full(sym, from_date_str=dm.DEFAULT_FROM_DATE, to_date_str=None, max_retries=3):
        assert sym == symbol
        return fake_full

    dm.fetch_vndirect_data = fetch_stub_full
    result = dm.update_stock_data(conn, symbol)
    assert result.action == "full_pull", result
    assert result.rows_after == len(fake_full), (result.rows_after, len(fake_full))
    print("[OK] Test 1 - full pull khi chua co du lieu:", result.message)

    # --- Test 2: incremental update -> xoa dong cuoi, keo lai tu ngay do den nay ---
    last_date_before = dm.get_last_date(conn, symbol)
    rows_before = dm.get_row_count(conn, symbol)

    # gia lap: ngay cuoi truoc do gia EOD sai (chua chot), va co them 2 ngay moi
    extended_end = today.strftime("%Y-%m-%d")
    fake_incremental = make_fake_df(last_date_before, extended_end)
    # sua gia cua ngay last_date_before de kiem tra co bi ghi de (upsert) khong
    fake_incremental.loc[pd.Timestamp(last_date_before), "close"] = 999.99

    def fetch_stub_incremental(sym, from_date_str=dm.DEFAULT_FROM_DATE, to_date_str=None, max_retries=3):
        assert from_date_str == last_date_before, (from_date_str, last_date_before)
        return fake_incremental

    dm.fetch_vndirect_data = fetch_stub_incremental
    result2 = dm.update_stock_data(conn, symbol)
    assert result2.action == "incremental_update", result2
    rows_after = dm.get_row_count(conn, symbol)
    assert rows_after > rows_before, (rows_after, rows_before)

    # kiem tra gia da duoc ghi de dung (upsert, khong bi trung key loi)
    df_check = dm.get_price_df(conn, symbol)
    updated_close = df_check.loc[pd.Timestamp(last_date_before), "close"]
    assert abs(updated_close - 999.99) < 1e-6, updated_close
    print(f"[OK] Test 2 - incremental update (xoa dong cuoi {last_date_before}, upsert lai):", result2.message)

    # kiem tra khong co duplicate primary key (symbol, date)
    dup_check = conn.execute(
        "SELECT symbol, date, COUNT(*) c FROM daily_prices GROUP BY symbol, date HAVING c > 1"
    ).fetchall()
    assert len(dup_check) == 0, dup_check
    print("[OK] Test 3 - khong co duplicate (symbol, date) trong DB")

    # --- Test 4: full_refresh -> xoa het, tai lai tu dau ---
    fake_refresh = make_fake_df("2026-01-01", extended_end)
    fake_refresh["close"] = fake_refresh["close"] + 1000  # gia khac han de nhan biet du lieu moi

    def fetch_stub_refresh(sym, from_date_str=dm.DEFAULT_FROM_DATE, to_date_str=None, max_retries=3):
        assert from_date_str == dm.DEFAULT_FROM_DATE
        return fake_refresh

    dm.fetch_vndirect_data = fetch_stub_refresh
    result3 = dm.full_refresh(conn, symbol)
    assert result3.action == "full_refresh", result3
    df_after_refresh = dm.get_price_df(conn, symbol)
    assert df_after_refresh["close"].min() > 900, df_after_refresh["close"].min()
    assert len(df_after_refresh) == len(fake_refresh)
    print("[OK] Test 4 - full_refresh xoa het va tai lai tu dau dung:", result3.message)

    # --- Test 5: error khi fetch tra ve rong (mo phong mat mang) ---
    def fetch_stub_empty(sym, from_date_str=dm.DEFAULT_FROM_DATE, to_date_str=None, max_retries=3):
        return pd.DataFrame()

    rows_before_err = dm.get_row_count(conn, symbol)
    dm.fetch_vndirect_data = fetch_stub_empty
    result4 = dm.update_stock_data(conn, symbol)
    assert result4.action == "error", result4
    rows_after_err = dm.get_row_count(conn, symbol)
    # chi mat 1 dong (dong cuoi da xoa de chuan bi refetch), khong mat toan bo du lieu
    assert rows_before_err - rows_after_err == 1, (rows_before_err, rows_after_err)
    print("[OK] Test 5 - fetch loi khong lam mat du lieu cu (chi mat 1 dong dang cho refetch):", result4.message)

    conn.close()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
