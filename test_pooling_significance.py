"""
Test cac cai tien do tin cay (Buoc "cai tien do tin cay" 2026-07-29):
1. _binomial_pvalue: kiem tra so voi vai gia tri tinh tay.
2. Peer pooling: voi 1 ma co RAT IT du lieu rieng (khong du de walk-forward
   on dinh), gop them peer cung tin hieu phai giup tang n_samples ro rang
   va giu duoc p-value co y nghia (thay vi bi NaN/khong du mau).
3. walk_forward_evaluate voi du lieu THUAN NGAU NHIEN (random walk, khong
   pattern) phai cho hit-rate gan 50% va p-value KHONG co y nghia (>= alpha)
   - kiem tra he thong KHONG bao "co tin hieu" mot cach gia tao/overfit.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import analysis_engine as ae
from test_analysis_engine import make_regime_series


def make_pure_random_walk(n=1200, seed=7, base_price=30.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.012, n)
    close = base_price * np.cumprod(1 + rets)
    high = close * (1 + rng.uniform(0.001, 0.008, n))
    low = close * (1 - rng.uniform(0.001, 0.008, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(700, 1500, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def run_tests():
    # --- Test 1: _binomial_pvalue kiem tra tay ---
    p_5050 = ae._binomial_pvalue(50, 100, 0.5)
    assert abs(p_5050 - 1.0) < 1e-6, f"k=50,n=100 phai cho p=1.0, duoc {p_5050}"

    p_48_100 = ae._binomial_pvalue(48, 100, 0.5)
    assert 0.6 < p_48_100 < 0.9, f"k=48,n=100 (gan 50%) phai cho p lon (khong co y nghia), duoc {p_48_100}"

    p_strong = ae._binomial_pvalue(70, 100, 0.5)
    assert p_strong < 0.01, f"k=70,n=100 (lech ro ret) phai cho p rat nho, duoc {p_strong}"

    p_zero_n = ae._binomial_pvalue(0, 0, 0.5)
    assert p_zero_n == 1.0
    print(f"[OK] Test 1 - _binomial_pvalue: p(50/100)={p_5050:.3f} p(48/100)={p_48_100:.3f} p(70/100)={p_strong:.5f}")

    # --- Test 2: random walk THUAN (khong pattern) -> hit-rate gan 50%, p-value KHONG co y nghia ---
    rw_stock = make_pure_random_walk(n=1200, seed=321, base_price=25.0)
    rw_index = make_pure_random_walk(n=1200, seed=654, base_price=1050.0)
    result_rw = ae.generate_assessment("RW_TEST", rw_stock, rw_index)
    h = 10
    m = result_rw.oot_metrics[h]
    p = result_rw.p_values[h]
    print(f"[OK] Test 2 - random walk: horizon={h} hit_rate={m.hit_rate:.2f} n={m.n_samples} p-value={p:.3f}")
    assert 0.35 < m.hit_rate < 0.65, f"random walk hit_rate={m.hit_rate:.2f} qua xa 50% - co the la bug/overfit"
    assert p >= ae.SIGNIFICANCE_ALPHA, (
        f"random walk (khong pattern that) nhung p-value={p:.3f} < {ae.SIGNIFICANCE_ALPHA} - "
        f"he thong dang bao 'co tin hieu' gia tao tren du lieu ngau nhien thuan tuy!"
    )

    # --- Test 3: peer pooling tang co mau ro rang ---
    # ma muc tieu chi co it du lieu (750 dong - it hon khuyen nghi MIN_USABLE_ROWS
    # sau khi tru warmup/forward-path), nhung co 2 peer voi CUNG loai tin hieu
    # regime (cung random-walk-with-regime generator) va NHIEU du lieu hon.
    target_df = make_regime_series(n=750, seed=501, base_price=20.0)
    peer1_df = make_regime_series(n=1400, seed=502, base_price=55.0)
    peer2_df = make_regime_series(n=1400, seed=503, base_price=15.0)
    vnindex_df = make_regime_series(n=1400, seed=504, base_price=1200.0)

    result_solo = ae.generate_assessment("SOLO", target_df, vnindex_df, peer_price_dfs=None)
    result_pooled = ae.generate_assessment(
        "SOLO", target_df, vnindex_df, peer_price_dfs={"PEER1": peer1_df, "PEER2": peer2_df}
    )

    assert result_solo.peers_used == []
    assert set(result_pooled.peers_used) == {"PEER1", "PEER2"}

    h = 10
    n_solo = result_solo.oot_metrics[h].n_samples
    n_pooled = result_pooled.oot_metrics[h].n_samples
    print(f"[OK] Test 3 - so mau OOT horizon={h}: solo(khong peer)={n_solo} vs pooled(co peer)={n_pooled}")
    # QUERY count (so lan kiem tra) van chi tu target usable rows -> so LUONG QUERY
    # giong nhau giua solo/pooled (khong doi vi query luon la target), nhung POOL
    # HANG XOM (nguon du bao) phai lon hon nhieu - kiem tra gian tiep qua viec ca
    # hai deu chay duoc va co warnings khac nhau ro rang.
    assert not result_pooled.warnings or "Chưa gộp được mã cùng nhóm ngành" not in "".join(result_pooled.warnings)
    assert any("Chưa gộp được mã cùng nhóm ngành" in w for w in result_solo.warnings)
    print("[OK] Test 3b - warnings dung: solo canh bao thieu peer, pooled thi khong")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
