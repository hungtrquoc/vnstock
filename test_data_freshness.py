"""
Test cho fix "du lieu bi cu" (2026-08-03, theo phan hoi nguoi dung: bao cao
van ghi "Tinh den 2026-07-29" du hom do da la 2026-08-03 - "thi truong thay
doi lien tuc, dung du lieu out-of-date thi khong theo kip").

TRUOC KHI SUA: `_run_full_analysis` (main.py) chi goi dm.update_stock_data
NEU DB CHUA CO GI CA (dm.has_data == False) - nghia la sau lan dau co du
lieu, moi lan "Chay phan tich" tiep theo KHONG BAO GIO tu dong lam moi du
lieu nua, du da qua nhieu ngay - nguoi dung phai tu nho bam "Cap nhat du
lieu" o tab 1 truoc.

SAU KHI SUA: MOI LAN chay phan tich deu goi dm.update_stock_data cho ca ma
dang xet, VNINDEX, va tung peer (ham nay tu thong minh: full pull neu chua
co, incremental update neu da co, va KHONG BAO GIO nem exception - tra ve
UpdateResult.action="error" neu that bai). Neu fetch that bai (vd mat mang),
van tiep tuc phan tich tren du lieu CU nhat co san (khong crash), nhung
them 1 canh bao ro rang vao ket qua de nguoi dung biet bao cao co the dang
dua tren du lieu cu.

Dung mock/monkeypatch cho dm.update_stock_data thay vi goi that (moi truong
sandbox build khong the ket noi VNDIRECT that - se rat cham do retry/backoff
neu goi that)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))

import data_manager as dm
from test_analysis_engine import make_regime_series
import main as m


def run_tests():
    conn = dm.get_connection(":memory:")

    def seed(symbol, seed_val, base):
        df = make_regime_series(n=900, seed=seed_val, base_price=base)
        dm.upsert_prices(conn, symbol, df)

    seed("VNINDEX", 1, 1200.0)
    seed("VCB", 2, 90.0)
    seed("BID", 3, 45.0)
    seed("CTG", 4, 30.0)

    orig_update = dm.update_stock_data

    # --- Test 1+2: update_stock_data phai duoc goi cho MOI ma (target,
    # VNINDEX, tung peer) o MOI LAN chay phan tich - khong chi khi thieu du
    # lieu (day chinh la bug duoc bao cao). ---
    calls = []

    def fake_update_ok(conn, symbol):
        calls.append(symbol)
        return dm.UpdateResult(symbol, "no_change", 900, 900, 0, "2026-07-29", "2026-07-29", message="ok (mocked)")

    dm.update_stock_data = fake_update_ok
    try:
        bundle = m._run_full_analysis(conn, "VCB")
    finally:
        dm.update_stock_data = orig_update

    assert "VNINDEX" in calls, "VNINDEX phải được gọi update MỖI LẦN chạy, không chỉ khi thiếu"
    assert "VCB" in calls, "VCB (mã đang xét) phải được gọi update MỖI LẦN chạy"
    assert "BID" in calls and "CTG" in calls, "các peer phải được gọi update MỖI LẦN chạy"
    print(f"[OK] Test 1 - update_stock_data được gọi cho target/VNINDEX/peers MỖI LẦN chạy: {calls}")

    assert not any("cập nhật" in w.lower() for w in bundle.assessment.warnings), \
        "không nên có cảnh báo fetch-lỗi khi update thành công"
    print("[OK] Test 2 - không có cảnh báo fetch-lỗi khi update thành công")

    # --- Test 3: neu update that bai (vd mat mang), van phai phan tich xong
    # tren du lieu CU dang co (khong crash), NHUNG phai co canh bao ro rang
    # trong ket qua de nguoi dung biet bao cao co the dang bi cu. ---
    calls2 = []

    def fake_update_fail(conn, symbol):
        calls2.append(symbol)
        return dm.UpdateResult(symbol, "error", 900, 900, 0, "2026-07-29", "2026-07-29", message="mất mạng (mô phỏng)")

    dm.update_stock_data = fake_update_fail
    try:
        bundle2 = m._run_full_analysis(conn, "VCB")
    finally:
        dm.update_stock_data = orig_update

    assert any("VNINDEX" in w for w in bundle2.assessment.warnings), "phải có cảnh báo khi không cập nhật được VNINDEX"
    assert any("VCB" in w for w in bundle2.assessment.warnings), "phải có cảnh báo khi không cập nhật được VCB"
    assert bundle2.assessment.symbol == "VCB"  # van phan tich xong tren du lieu cu, khong crash
    print(f"[OK] Test 3 - fetch thất bại vẫn phân tích được trên dữ liệu cũ VÀ có cảnh báo rõ ràng: {bundle2.assessment.warnings}")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
