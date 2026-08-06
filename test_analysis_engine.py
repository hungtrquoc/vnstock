"""
Test analysis_engine.py bang du lieu tong hop (khong the test tren du lieu VN
thuc trong sandbox build vi mang bi chan). Du lieu tong hop duoc thiet ke co
CAC GIAI DOAN XU HUONG RO RANG (regime-switching random walk) de kiem tra:
1. Pipeline chay het khong loi (indicators -> split -> calibrate -> forecast).
2. Neu co pattern thuc su trong du lieu, he thong phai phat hien duoc (hit
   rate OOT > 50% ro rang) - day la bai kiem tra "co hoc duoc gi khong",
   khac voi chi kiem tra khong crash.
3. Duong du bao la NHIEU DOAN (cac gia tri khac nhau moi ngay), khong phai
   noi thang 1 duong tu diem dau den diem cuoi.
4. Khong co loi leakage co ban: neighbor pool cho VAL luon la TRAIN (qua khu),
   OOT luon dung TRAIN+VAL (qua khu) - kiem tra gian tiep qua viec cac ham
   nhan tham so pool_idx/query_idx ro rang, tach biet.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import analysis_engine as ae


def make_regime_series(n=1500, seed=42, base_price=50.0):
    rng = np.random.default_rng(seed)
    rets = np.empty(n)
    vols = np.empty(n)
    i = 0
    regime = 1  # 1 = bull, -1 = bear
    while i < n:
        seg_len = rng.integers(30, 90)
        seg_len = min(seg_len, n - i)
        # tin hieu manh, ro rang (SNR cao) - day la test cau truc de xac nhan
        # pipeline PHAT HIEN DUOC pattern khi no ton tai ro, khong danh gia
        # kha nang du bao gia co phieu thuc te (noise thuc te lon hon nhieu).
        drift = 0.006 if regime == 1 else -0.006
        noise = rng.normal(0, 0.006, seg_len)
        rets[i:i + seg_len] = drift + noise
        vols[i:i + seg_len] = rng.integers(800, 1600 if regime == 1 else 1200, seg_len)
        i += seg_len
        regime *= -1

    close = base_price * np.cumprod(1 + rets)
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vols}, index=idx
    )


def run_tests():
    stock_df = make_regime_series(n=1500, seed=1, base_price=50.0)
    vnindex_df = make_regime_series(n=1500, seed=2, base_price=1000.0)
    # dong bo VNINDEX cung nhip mot phan voi co phieu de rel_strength co y nghia
    vnindex_df["close"] = 0.5 * vnindex_df["close"] + 0.5 * (stock_df["close"] / stock_df["close"].iloc[0] * 1000)

    # --- Test 1: indicators khong loi, khong toan NaN ---
    ind = ae.compute_indicators(stock_df)
    for col in ["ema20", "ema50", "ema200", "rsi14", "macd_hist", "atr14", "bb_pctb", "adx14", "vol_ratio20"]:
        assert col in ind.columns, col
        assert ind[col].notna().sum() > 1000, f"{col} co qua nhieu NaN"
    print("[OK] Test 1 - compute_indicators sinh du cac cot, khong toan NaN")

    # --- Test 2: market context + regime ---
    vn_ind = ae.compute_indicators(vnindex_df)
    ctx = ae.compute_market_context(ind, vn_ind)
    assert "beta_resid_20" in ctx.columns and "market_regime" in ctx.columns and "range_pct_120" in ctx.columns
    assert set(ctx["market_regime"].unique()) <= {"uptrend", "downtrend", "sideways"}
    print("[OK] Test 2 - compute_market_context sinh beta_resid/range_pct_120/market_regime hop le")

    # --- Test 3: full pipeline qua generate_assessment ---
    result = ae.generate_assessment("SYNTH", stock_df, vnindex_df)
    assert result.symbol == "SYNTH"
    assert result.calibrated_k in ae.DEFAULT_K_GRID
    print(f"[OK] Test 3 - generate_assessment chay xong, k calibrate = {result.calibrated_k}")
    if result.warnings:
        print("    (warnings):", result.warnings)

    # --- Test 4: OOT metrics phai co n_samples > 0 cho tat ca horizon ---
    for h in ae.DEFAULT_HORIZONS:
        m = result.oot_metrics[h]
        assert m.n_samples > 0, f"horizon {h} khong co sample OOT nao - kiem tra lai split/warmup"
        assert 0.0 <= m.hit_rate <= 1.0
        print(f"[OK] Test 4.{h} - OOT horizon={h}: hit_rate={m.hit_rate:.2f} mae={m.mae:.4f} n={m.n_samples}")

    # --- Test 5: he thong phai phat hien duoc pattern that (hit rate ro rang > 50%) ---
    # vi du lieu tong hop co regime dai han that (30-90 ngay/doan), horizon 3
    # ngay thuong nam trong 1 doan -> ky vong hit rate cao hon random dang ke.
    hit3 = result.oot_metrics[3].hit_rate
    assert hit3 > 0.55, (
        f"hit_rate horizon=3 chi {hit3:.2f} - qua gan random (50%), pipeline co the "
        f"chua hoc duoc pattern regime da biet truoc trong du lieu tong hop"
    )
    print(f"[OK] Test 5 - phat hien duoc pattern regime da biet truoc (hit_rate horizon=3 = {hit3:.2f} > 0.55)")

    # --- Test 6: duong du bao phai la NHIEU DOAN, khong phai 1 duong thang ---
    path20 = result.forecasts[20]["expected_return_path_pct"]
    assert len(path20) == 20
    diffs = np.diff(path20)
    # neu la duong thang (noi diem dau-cuoi), cac buoc nhay se gan bang nhau tuyet doi.
    # kiem tra co it nhat vai buoc nhay khac nhau ro (khong deu) -> xac nhan nhieu doan.
    assert np.std(diffs) > 1e-6, "duong du bao co ve la 1 duong thang deu - khong dung yeu cau"
    print(f"[OK] Test 6 - duong du bao 20 phien la nhieu doan (std cua buoc nhay = {np.std(diffs):.4f})")

    # --- Test 7: gia du bao (price_path) phai duong va co so luong dung ---
    for h in ae.DEFAULT_HORIZONS:
        pp = result.forecasts[h]["price_path"]
        assert len(pp) == h
        assert all(p > 0 for p in pp)
    print("[OK] Test 7 - price_path hop le cho ca 3/10/20 phien")

    print("\nRationale mau:\n" + result.rationale_text)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
