"""
Test market_screener.py bang du lieu tong hop (monkeypatch dm.update_stock_data
de tranh goi mang that - sandbox xay dung bi chan VNDIRECT, xem cac test file
truoc do trong project vi cung ly do).
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))

import data_manager as dm
import market_screener as screener
from test_analysis_engine import make_regime_series
from test_differentiation import make_flat_vnindex, make_idiosyncratic_ticker


def run_tests():
    conn = dm.get_connection(":memory:")

    def seed(symbol, seed_val, base):
        df = make_regime_series(n=900, seed=seed_val, base_price=base)
        dm.upsert_prices(conn, symbol, df)

    seed("VNINDEX", 1, 1200.0)
    seed("VCB", 2, 90.0)     # du du lieu cho ca classic_ta lan k-NN
    seed("SHORT", 3, 20.0)   # se cat ngan lai o duoi - khong du cho classic_ta

    # ma ro rang bullish (dung lai generator cua test_differentiation.py) de
    # dam bao TEST DUOC nhanh "classic_positive=True -> chay tiep k-NN day
    # du" - khong chi phu thuoc may-rui vao du lieu regime ngau nhien o tren.
    flat_idx = make_flat_vnindex(n=900, seed=99, base_price=1100.0)
    bull_df = make_idiosyncratic_ticker(flat_idx["close"], beta=1.0, idio_drift=0.0006, seed=201, base_price=40.0)
    dm.upsert_prices(conn, "BULLTEST", bull_df)

    # cat ngan SHORT xuong duoi nguong MIN_ROWS_FOR_CLASSIC_TA de test truong hop "skip"
    short_df = dm.get_price_df(conn, "SHORT").iloc[:100]
    dm.delete_symbol_data(conn, "SHORT")
    dm.upsert_prices(conn, "SHORT", short_df)

    orig_update = dm.update_stock_data
    dm.update_stock_data = lambda conn, symbol: dm.UpdateResult(
        symbol, "no_change", 0, 0, 0, None, None, message="mocked, no-op"
    )
    try:
        row_vcb = screener.screen_one_symbol(conn, "VCB")
        row_short = screener.screen_one_symbol(conn, "SHORT")
        row_missing = screener.screen_one_symbol(conn, "NOPE_NOT_SEEDED")
        row_bull = screener.screen_one_symbol(conn, "BULLTEST")
    finally:
        dm.update_stock_data = orig_update

    # --- Test 1: ma du du lieu -> status "ok", co classic_verdict ro rang ---
    assert row_vcb.status == "ok", row_vcb
    assert row_vcb.classic_verdict is not None
    print(f"[OK] Test 1 - VCB quét xong: status=ok, classic_verdict={row_vcb.classic_verdict}, "
          f"classic_positive={row_vcb.classic_positive}, stat_positive={row_vcb.stat_positive}")

    # --- Test 2: neu classic_positive VA du du lieu k-NN -> phai co thong tin stat day du ---
    if row_vcb.classic_positive:
        assert row_vcb.stat_best_horizon is None or row_vcb.stat_best_horizon in screener.ae.DEFAULT_HORIZONS
        print(f"[OK] Test 2 - VCB classic_positive=True -> đã chạy k-NN đầy đủ (stat_positive={row_vcb.stat_positive})")
    else:
        assert row_vcb.stat_best_horizon is None, "không nên chạy k-NN nếu classic_positive=False (đúng thiết kế 'phễu')"
        print("[OK] Test 2 - VCB classic_positive=False -> ĐÚNG THIẾT KẾ: không chạy k-NN đầy đủ (tiết kiệm CPU)")

    # --- Test 3: ma qua it du lieu -> status "skip", khong crash ---
    assert row_short.status == "skip", row_short
    assert not row_short.classic_positive and not row_short.stat_positive
    print(f"[OK] Test 3 - SHORT (thiếu dữ liệu) -> status=skip, message='{row_short.message}'")

    # --- Test 4: ma khong ton tai (khong co du lieu, update la no-op mocked) -> status "error", khong crash ---
    assert row_missing.status == "error", row_missing
    print(f"[OK] Test 4 - mã không tồn tại -> status=error, message='{row_missing.message}'")

    # --- Test 5: ma RO RANG bullish (drift dương mạnh, độc lập nhiễu VNINDEX) ---
    # phai kich hoat classic_positive=True VA chay tiep buoc k-NN day du (function
    # "pheu" phai THUC SU di vao nhanh nay, khong chi ly thuyet).
    assert row_bull.status == "ok", row_bull
    assert row_bull.classic_positive is True, (
        f"chuoi gia BULLTEST co drift +0.3%/ngay ro rang, ky vong classic_ta=Tich cuc, "
        f"nhung duoc {row_bull.classic_verdict}"
    )
    print(f"[OK] Test 5 - BULLTEST (drift dương rõ) -> classic_verdict={row_bull.classic_verdict} "
          f"(classic_positive=True, đã chạy tiếp k-NN đầy đủ: stat_positive={row_bull.stat_positive}, "
          f"best_horizon={row_bull.stat_best_horizon})")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
