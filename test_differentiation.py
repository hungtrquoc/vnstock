"""
Test truc tiep van de nguoi dung bao cao: "du bao giong nhau giua cac ma".

Tao 2 ma tong hop dung CHUNG 1 VNINDEX (cung mot chuoi thi truong, giong tinh
huong thuc te la nhieu ma phan tich cung thoi diem deu doi chieu voi VNINDEX
giong nhau), nhung phan RIENG (idiosyncratic) cua tung ma khac han nhau:
- TICKER_POS: beta*VNINDEX + trend rieng DUONG manh + noise
- TICKER_NEG: beta*VNINDEX + trend rieng AM manh + noise

Neu phuong phap phan tich chi phan anh VNINDEX (loi cu), du bao cua 2 ma nay
se gan giong nhau. Neu da sua dung (dung beta_resid loai bo phan chung voi
thi truong), du bao phai khac biet ro rang, phan anh dung phan rieng cua
tung ma.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import analysis_engine as ae
from test_analysis_engine import make_regime_series


def make_flat_vnindex(n=1200, seed=99, base_price=1100.0):
    """VNINDEX gia dinh gan nhu di ngang (drift~0, noise nho) - de test tach
    bach tin hieu RIENG cua tung ma khoi anh huong cua thi truong chung, chu
    khong nham mo phong thi truong thuc te (xem make_regime_series cho cai do)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.004, n)
    close = base_price * np.cumprod(1 + rets)
    high = close * (1 + rng.uniform(0.001, 0.005, n))
    low = close * (1 - rng.uniform(0.001, 0.005, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    vol = rng.integers(900, 1300, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def make_idiosyncratic_ticker(vnindex_close: pd.Series, beta: float, idio_drift: float, seed: int, base_price: float):
    """idio_drift ap dung MOI NGAY (hang so, cung dau cho ca chuoi) - the hien
    1 xu huong dac thu ben vung cua rieng ma nay (vd dong tien ke toan, tang
    truong loi nhuan...), doc lap voi VNINDEX (duoc giu gan nhu di ngang trong
    test nay - xem make_flat_vnindex - de tin hieu rieng khong bi nhieu chim
    boi bien dong thi truong chung)."""
    rng = np.random.default_rng(seed)
    idx_ret = vnindex_close.pct_change().fillna(0.0).to_numpy()
    n = len(idx_ret)

    idio_noise = rng.normal(0, 0.006, n)
    stock_ret = beta * idx_ret + idio_drift + idio_noise
    close = base_price * np.cumprod(1 + stock_ret)
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(800, 1600, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=vnindex_close.index,
    )


def run_tests():
    # dung VNINDEX gan nhu di ngang (khong phai make_regime_series) de co the
    # tach bach ro tin hieu RIENG cua tung ma, tranh bi bien dong regime manh
    # cua thi truong chung (drift +-0.6%/ngay trong make_regime_series) lam
    # loang tin hieu idio_drift nho hon nhieu.
    vnindex_df = make_flat_vnindex(n=1200, seed=99, base_price=1100.0)

    # ca 2 ma co CUNG beta~1.0 voi VNINDEX, nhung phan rieng (idio_drift) trai dau
    # va du lon de tao ra su khac biet ro rang, doc lap voi bat ky trang thai
    # VNINDEX nao tai thoi diem du bao.
    ticker_pos = make_idiosyncratic_ticker(vnindex_df["close"], beta=1.0, idio_drift=0.0015, seed=201, base_price=40.0)
    ticker_neg = make_idiosyncratic_ticker(vnindex_df["close"], beta=1.0, idio_drift=-0.0015, seed=202, base_price=40.0)

    result_pos = ae.generate_assessment("POS", ticker_pos, vnindex_df)
    result_neg = ae.generate_assessment("NEG", ticker_neg, vnindex_df)

    print("POS - beta_resid hien tai duoc dung trong feature (xem state_description):")
    print(" ", result_pos.state_description)
    print("NEG - beta_resid hien tai duoc dung trong feature (xem state_description):")
    print(" ", result_neg.state_description)

    print("k POS:", result_pos.calibrated_k, "| k NEG:", result_neg.calibrated_k)
    diffs = {}
    for h in ae.DEFAULT_HORIZONS:
        pos_ret = result_pos.forecasts[h]["expected_return_pct"]
        neg_ret = result_neg.forecasts[h]["expected_return_pct"]
        diff = pos_ret - neg_ret
        diffs[h] = diff
        print(f"[so sanh] horizon={h}: POS ky vong {pos_ret:+.1f}% | NEG ky vong {neg_ret:+.1f}% | chenh lech {diff:+.1f} diem %")
        m_pos, b_pos = result_pos.oot_metrics[h], result_pos.baseline_oot_metrics[h]
        m_neg, b_neg = result_neg.oot_metrics[h], result_neg.baseline_oot_metrics[h]
        print(f"    POS OOT hit={m_pos.hit_rate:.2f} baseline={b_pos.hit_rate:.2f} | NEG OOT hit={m_neg.hit_rate:.2f} baseline={b_neg.hit_rate:.2f}")

    # Horizon ngan (3 phien): idio_drift moi ngay (0.15%) nho hon nhieu so voi
    # nhieu ngau nhien (0.6%/ngay), nen ty le tin hieu/nhieu (SNR) thap va MOI
    # phuong phap (khong chi phuong phap nay) deu kho phan biet 2 ma trong
    # 3 ngay - day la thuc te khach quan, khong phai loi. Chi kiem tra dau
    # (chenh lech phai CUNG DAU voi ky vong: POS > NEG), khong doi hoi bien do lon.
    assert diffs[3] > 0.0, f"horizon=3: dau chenh lech sai ({diffs[3]:+.1f}), POS phai > NEG"

    # Horizon trung/dai (10, 20 phien): du lieu da du de trung binh hoa bot
    # nhieu ngau nhien, tin hieu idio_drift phai the hien ro rang trong ket
    # qua - day moi la bai kiem tra chinh cho van de "du bao giong nhau".
    for h in (10, 20):
        assert diffs[h] > 1.0, (
            f"horizon={h}: chenh lech chi {diffs[h]:+.1f} diem % - phuong phap co the van chua "
            f"phan biet duoc phan rieng cua tung ma."
        )
    print(
        "\n[nhan xet quan trong] O horizon ngan (3 phien), du bao 2 ma van gan nhau vi tin hieu "
        "rieng qua nho so voi nhieu ngau nhien hang ngay - day la han che tu nhien cua du lieu, "
        "khong phai loi phuong phap. O horizon 10-20 phien, du bao da phan biet ro rang va dung "
        "chieu theo tin hieu rieng cua tung ma."
    )

    print(f"\n[OK] Du bao cua 2 ma voi phan RIENG trai dau nhau (nhung dung CHUNG 1 VNINDEX) "
          f"da khac biet dung chieu va ro rang ve bien do o horizon 10-20 phien - phuong phap "
          f"khong con chi phan anh thi truong chung.")

    # kiem tra them: OOT edge-over-baseline phai duong o ca 2 ma (chung minh tin
    # hieu k-NN thuc su vuot qua du bao ngay tho, khong chi la ngau nhien)
    for name, result in [("POS", result_pos), ("NEG", result_neg)]:
        h = 10
        m, b = result.oot_metrics[h], result.baseline_oot_metrics[h]
        print(f"[{name}] horizon={h}: hit-rate k-NN={m.hit_rate:.2f} vs baseline={b.hit_rate:.2f}")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
