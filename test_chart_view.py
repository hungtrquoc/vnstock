"""
Test chart_view.py bang backend Agg (khong can PyQt/display). Kiem tra ham
plot_analysis khong loi va thuc su ve duong du bao NHIEU DOAN (nhieu diem
khac gia tri), co dai tin cay (fill_between), co cac moc 3/10/20 phien.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import analysis_engine as ae
import chart_view as cv
from test_analysis_engine import make_regime_series


def run_tests():
    stock_df = make_regime_series(n=900, seed=11, base_price=42.0)
    vnindex_df = make_regime_series(n=900, seed=12, base_price=1100.0)

    result = ae.generate_assessment("CHARTTEST", stock_df, vnindex_df)

    fig, ax = plt.subplots(figsize=(9, 5))
    cv.plot_analysis(ax, "CHARTTEST", stock_df, result, lookback_days=120)

    # --- Test 1: co duong gia lich su + duong du bao ---
    assert len(ax.lines) >= 2, "thieu duong ve (gia lich su / du bao)"
    print(f"[OK] Test 1 - co {len(ax.lines)} duong duoc ve tren chart")

    # --- Test 2: duong du bao la nhieu doan (nhieu diem, gia tri khac nhau) ---
    forecast_line = None
    for line in ax.lines:
        if line.get_label().startswith("D") and "báo" in line.get_label():
            forecast_line = line
            break
    assert forecast_line is not None, "khong tim thay duong du bao trong legend"
    ydata = forecast_line.get_ydata()
    assert len(ydata) >= 4, f"duong du bao chi co {len(ydata)} diem - qua it de la 'nhieu doan'"
    diffs = [ydata[i + 1] - ydata[i] for i in range(len(ydata) - 1)]
    assert len(set(round(d, 6) for d in diffs)) > 1, "cac buoc nhay deu nhau - co ve la 1 duong thang"
    print(f"[OK] Test 2 - duong du bao co {len(ydata)} diem, cac buoc nhay khac nhau (nhieu doan thuc su)")

    # --- Test 3: co dai tin cay (fill_between) ---
    assert len(ax.collections) >= 1, "khong co vung fill_between (dai tin cay)"
    print(f"[OK] Test 3 - co {len(ax.collections)} vung fill_between (dai tin cay)")

    # --- Test 4: co it nhat 3 duong axvline moc (hien tai + 3/10/20 phien) ---
    # axvline cung la Line2D nen nam trong ax.lines; kiem tra so luong duong tong the >= 6
    # (2 gia lich su/ema + 1 du bao + it nhat 3 moc thoi gian)
    assert len(ax.lines) >= 6, f"qua it duong ve ({len(ax.lines)}) - co the thieu cac moc 3/10/20 phien"
    print(f"[OK] Test 4 - tong {len(ax.lines)} duong (bao gom cac moc thoi gian)")

    # --- Test 5: co annotation (nhan moc % + mui ten) ---
    assert len(ax.texts) >= 3, f"qua it annotation text ({len(ax.texts)}) - thieu nhan % cho 3/10/20 phien"
    print(f"[OK] Test 5 - co {len(ax.texts)} annotation text (nhan % cac moc)")

    out_path = os.path.join(os.path.dirname(__file__), "_test_chart_output.png")
    fig.savefig(out_path, dpi=100)
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 1000
    print(f"[OK] Test 6 - da luu chart ra {out_path} (kich thuoc {os.path.getsize(out_path)} bytes)")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
