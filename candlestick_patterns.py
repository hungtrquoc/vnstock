"""
candlestick_patterns.py
------------------------
Nhan dien mo hinh nen Nhat (candlestick) tren du lieu OHLC. Module DOC LAP,
khong phu thuoc analysis_engine.py/data_manager.py (giu nguyen kien truc tach
biet data/analysis/chart/UI cua project).

Boi canh (theo yeu cau nguoi dung 2026-07-29): "danh gia dua tren cac chi bao
ky thuat, bo sung them cac mo hinh nen de danh gia va ve mo hinh nen dang
trung vao bieu do". Day la lop danh gia THEO KINH NGHIEM/DINH TINH (classic
technical analysis), KHAC voi phan du bao k-NN da kiem chung OOT trong
analysis_engine.py - xem classic_ta.py de biet cach 2 lop nay duoc trinh bay
rieng ret, khong lam nguoi dung nham lan giua "co kiem chung thong ke" va
"chi la kinh nghiem/quy uoc bieu do", dung lap lai sai lam cua cac notebook cu
(xem memory vn_stock_prior_attempts_findings - trinh bay diem so dinh tinh nhu
the la ket qua model da kiem chung).

Cac mo hinh duoc nhan dien (13 mo hinh chuan):
  1 nen : Doji, Hammer, Inverted Hammer, Shooting Star, Hanging Man
  2 nen : Bullish Engulfing, Bearish Engulfing, Bullish Harami, Bearish Harami,
          Piercing Line, Dark Cloud Cover
  3 nen : Morning Star, Evening Star, Three White Soldiers, Three Black Crows

Nguyen tac quan trong: cac mo hinh dao chieu (Hammer/Hanging Man,
Inverted Hammer/Shooting Star, Engulfing, Harami, Piercing/Dark Cloud,
Morning/Evening Star) CAN XET XU HUONG TRUOC DO (prior trend) de phan loai
dung - hinh dang nen giong nhau nhung Y NGHIA khac nhau tuy xuat hien sau
xu huong tang hay giam. Three White Soldiers/Three Black Crows la tin hieu
TIEP DIEN/xac nhan manh, khong can dieu kien xu huong truoc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cac ham do luong hinh dang nen co ban
# ---------------------------------------------------------------------------

def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    r = df["high"] - df["low"]
    return r.replace(0, np.nan)


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def _is_bullish(df: pd.DataFrame) -> pd.Series:
    return df["close"] > df["open"]


def _prior_trend(close: pd.Series, i: int, lookback: int = 10, thresh: float = 0.02) -> str:
    """Xu huong TRUOC vi tri i (khong bao gom nen i), dua tren % thay doi gia
    tu i-lookback-1 den i-1. Tra ve 'up' / 'down' / 'sideways'. Dung nguong
    tuong doi don gian (khong phai regression) - du dung de phan loai dung
    nhom mo hinh (hammer vs hanging man...), khong can chinh xac tuyet doi."""
    start = i - lookback - 1
    end = i - 1
    if start < 0 or end < 0 or end >= len(close):
        return "sideways"
    base = close.iloc[start]
    if base == 0 or np.isnan(base):
        return "sideways"
    chg = close.iloc[end] / base - 1.0
    if chg > thresh:
        return "up"
    if chg < -thresh:
        return "down"
    return "sideways"


@dataclass
class PatternHit:
    index: int          # vi tri (iloc) cua nen "tin hieu" (nen cuoi cung trong mo hinh nhieu nen)
    name: str           # ten mo hinh (tieng Viet)
    direction: str      # 'bullish' | 'bearish'
    n_candles: int      # so nen tao thanh mo hinh (1/2/3)


# ---------------------------------------------------------------------------
# Nhan dien mo hinh 1 nen
# ---------------------------------------------------------------------------

def _detect_single_candle(df: pd.DataFrame, lookback_trend: int = 10) -> list[PatternHit]:
    hits: list[PatternHit] = []
    body = _body(df)
    rng = _range(df)
    up_sh = _upper_shadow(df)
    low_sh = _lower_shadow(df)
    body_pct = (body / rng).fillna(0.0)

    for i in range(len(df)):
        b, r = body.iloc[i], rng.iloc[i]
        if pd.isna(r) or r == 0:
            continue
        bp = body_pct.iloc[i]
        us, ls = up_sh.iloc[i], low_sh.iloc[i]

        # Doji: than nen rat nho so voi bien do (<=10%)
        if bp <= 0.10:
            hits.append(PatternHit(i, "Doji (lung lay/do du)", "neutral", 1))
            continue  # doji khong dong thoi la hammer/shooting star

        # Hinh dang "bua/sao doi chieu": than nho (<=35% bien do), 1 bong dai
        # (>=2x than), bong con lai rat ngan (<=than).
        small_body = bp <= 0.35
        if not small_body:
            continue

        long_lower = ls >= 2 * b and us <= b
        long_upper = us >= 2 * b and ls <= b
        trend = _prior_trend(df["close"], i, lookback_trend)

        if long_lower:
            if trend == "down":
                hits.append(PatternHit(i, "Hammer (bua - đảo chiều tăng)", "bullish", 1))
            elif trend == "up":
                hits.append(PatternHit(i, "Hanging Man (người treo cổ - đảo chiều giảm)", "bearish", 1))
        elif long_upper:
            if trend == "down":
                hits.append(PatternHit(i, "Inverted Hammer (bua ngược - đảo chiều tăng)", "bullish", 1))
            elif trend == "up":
                hits.append(PatternHit(i, "Shooting Star (sao đổi ngôi - đảo chiều giảm)", "bearish", 1))

    return hits


# ---------------------------------------------------------------------------
# Nhan dien mo hinh 2 nen
# ---------------------------------------------------------------------------

def _detect_two_candle(df: pd.DataFrame, lookback_trend: int = 10) -> list[PatternHit]:
    hits: list[PatternHit] = []
    body = _body(df)
    o, c = df["open"], df["close"]
    bullish = _is_bullish(df)

    for i in range(1, len(df)):
        p, cur = i - 1, i
        trend = _prior_trend(df["close"], p, lookback_trend)
        b_prev, b_cur = body.iloc[p], body.iloc[cur]
        if b_prev == 0 or pd.isna(b_prev):
            continue

        # --- Engulfing: nen sau "nuot tron" than nen truoc, doi mau ---
        if (not bullish.iloc[p]) and bullish.iloc[cur]:
            if c.iloc[cur] >= o.iloc[p] and o.iloc[cur] <= c.iloc[p] and trend != "up":
                hits.append(PatternHit(cur, "Bullish Engulfing (nhấn chìm tăng)", "bullish", 2))
        if bullish.iloc[p] and (not bullish.iloc[cur]):
            if o.iloc[cur] >= c.iloc[p] and c.iloc[cur] <= o.iloc[p] and trend != "down":
                hits.append(PatternHit(cur, "Bearish Engulfing (nhấn chìm giảm)", "bearish", 2))

        # --- Harami: nen sau nho, nam TRONG than nen truoc, doi mau ---
        if (not bullish.iloc[p]) and bullish.iloc[cur] and trend == "down":
            if o.iloc[cur] >= c.iloc[p] and c.iloc[cur] <= o.iloc[p] and b_cur < b_prev * 0.7:
                hits.append(PatternHit(cur, "Bullish Harami (harami tăng)", "bullish", 2))
        if bullish.iloc[p] and (not bullish.iloc[cur]) and trend == "up":
            if c.iloc[cur] >= o.iloc[p] and o.iloc[cur] <= c.iloc[p] and b_cur < b_prev * 0.7:
                hits.append(PatternHit(cur, "Bearish Harami (harami giảm)", "bearish", 2))

        # --- Piercing Line: giam mạnh, nen sau mở gap xuong nhung dong cua
        #     tren 50% than nen truoc (khong vuot dinh) ---
        if trend == "down" and (not bullish.iloc[p]) and bullish.iloc[cur]:
            mid_prev = (o.iloc[p] + c.iloc[p]) / 2
            if o.iloc[cur] < c.iloc[p] and mid_prev < c.iloc[cur] < o.iloc[p]:
                hits.append(PatternHit(cur, "Piercing Line (xuyên thấu - đảo chiều tăng)", "bullish", 2))

        # --- Dark Cloud Cover: tang mạnh, nen sau mo gap len nhung dong cua
        #     duoi 50% than nen truoc ---
        if trend == "up" and bullish.iloc[p] and (not bullish.iloc[cur]):
            mid_prev = (o.iloc[p] + c.iloc[p]) / 2
            if o.iloc[cur] > c.iloc[p] and c.iloc[p] < c.iloc[cur] < mid_prev:
                hits.append(PatternHit(cur, "Dark Cloud Cover (mây đen che phủ - đảo chiều giảm)", "bearish", 2))

    return hits


# ---------------------------------------------------------------------------
# Nhan dien mo hinh 3 nen
# ---------------------------------------------------------------------------

def _detect_three_candle(df: pd.DataFrame, lookback_trend: int = 10) -> list[PatternHit]:
    hits: list[PatternHit] = []
    body = _body(df)
    o, c = df["open"], df["close"]
    bullish = _is_bullish(df)

    for i in range(2, len(df)):
        i0, i1, i2 = i - 2, i - 1, i
        trend = _prior_trend(df["close"], i0, lookback_trend)
        b0, b1, b2 = body.iloc[i0], body.iloc[i1], body.iloc[i2]
        if b0 == 0 or pd.isna(b0):
            continue

        # --- Morning Star: giam mạnh -> nen nho (do du) -> tang mạnh, dong
        #     cua nen 3 vuot qua trung diem nen 1 ---
        if trend == "down" and (not bullish.iloc[i0]) and bullish.iloc[i2]:
            small_middle = b1 < b0 * 0.5
            gaps_down = max(o.iloc[i1], c.iloc[i1]) < c.iloc[i0]
            closes_above_mid = c.iloc[i2] > (o.iloc[i0] + c.iloc[i0]) / 2
            if small_middle and closes_above_mid:
                hits.append(PatternHit(i2, "Morning Star (sao mai - đảo chiều tăng)", "bullish", 3))

        # --- Evening Star: tang mạnh -> nen nho -> giam mạnh, dong cua nen 3
        #     xuong duoi trung diem nen 1 ---
        if trend == "up" and bullish.iloc[i0] and (not bullish.iloc[i2]):
            small_middle = b1 < b0 * 0.5
            closes_below_mid = c.iloc[i2] < (o.iloc[i0] + c.iloc[i0]) / 2
            if small_middle and closes_below_mid:
                hits.append(PatternHit(i2, "Evening Star (sao hôm - đảo chiều giảm)", "bearish", 3))

        # --- Three White Soldiers: 3 nen tang lien tiep, mo trong than nen
        #     truoc, dong cua ngay cang cao, than tuong doi dai ---
        if bullish.iloc[i0] and bullish.iloc[i1] and bullish.iloc[i2]:
            rng0, rng1, rng2 = _range(df).iloc[i0], _range(df).iloc[i1], _range(df).iloc[i2]
            if all(not pd.isna(x) and x > 0 for x in (rng0, rng1, rng2)):
                long_bodies = (b0 / rng0 > 0.55) and (b1 / rng1 > 0.55) and (b2 / rng2 > 0.55)
                rising_closes = c.iloc[i1] > c.iloc[i0] and c.iloc[i2] > c.iloc[i1]
                opens_inside = (o.iloc[i1] > o.iloc[i0]) and (o.iloc[i1] < c.iloc[i0]) and \
                               (o.iloc[i2] > o.iloc[i1]) and (o.iloc[i2] < c.iloc[i1])
                if long_bodies and rising_closes and opens_inside:
                    hits.append(PatternHit(i2, "Three White Soldiers (ba chàng lính trắng - xác nhận tăng)", "bullish", 3))

        # --- Three Black Crows: 3 nen giam lien tiep, tuong tu nhu tren
        #     nhung nguoc chieu ---
        if (not bullish.iloc[i0]) and (not bullish.iloc[i1]) and (not bullish.iloc[i2]):
            rng0, rng1, rng2 = _range(df).iloc[i0], _range(df).iloc[i1], _range(df).iloc[i2]
            if all(not pd.isna(x) and x > 0 for x in (rng0, rng1, rng2)):
                long_bodies = (b0 / rng0 > 0.55) and (b1 / rng1 > 0.55) and (b2 / rng2 > 0.55)
                falling_closes = c.iloc[i1] < c.iloc[i0] and c.iloc[i2] < c.iloc[i1]
                opens_inside = (o.iloc[i1] < o.iloc[i0]) and (o.iloc[i1] > c.iloc[i0]) and \
                               (o.iloc[i2] < o.iloc[i1]) and (o.iloc[i2] > c.iloc[i1])
                if long_bodies and falling_closes and opens_inside:
                    hits.append(PatternHit(i2, "Three Black Crows (ba con quạ đen - xác nhận giảm)", "bearish", 3))

    return hits


# ---------------------------------------------------------------------------
# Ham chinh
# ---------------------------------------------------------------------------

def detect_patterns(df: pd.DataFrame, lookback_trend: int = 10) -> list[PatternHit]:
    """Nhan dien tat ca mo hinh nen tren DataFrame OHLC (index = ngay, cot
    open/high/low/close). Tra ve list PatternHit, sap xep theo index tang dan.
    An toan voi thoi gian: moi mo hinh tai vi tri i chi dung du lieu <= i,
    khong nhin ve tuong lai (khong co rui ro leakage)."""
    hits = (
        _detect_single_candle(df, lookback_trend)
        + _detect_two_candle(df, lookback_trend)
        + _detect_three_candle(df, lookback_trend)
    )
    hits.sort(key=lambda h: h.index)
    return hits


def recent_patterns(df: pd.DataFrame, n_recent: int = 10, lookback_trend: int = 10) -> list[dict]:
    """Tra ve cac mo hinh nen xay ra trong N nen GAN NHAT (de hien thi/ve
    chu thich tren bieu do - dung cho yeu cau 've mo hinh nen dang trung vao
    bieu do'). Moi phan tu la dict co index/date/name/direction/n_candles."""
    hits = detect_patterns(df, lookback_trend)
    cutoff = len(df) - n_recent
    out = []
    for h in hits:
        if h.index >= cutoff:
            date = df.index[h.index]
            out.append({
                "index": h.index,
                "date": date,
                "name": h.name,
                "direction": h.direction,
                "n_candles": h.n_candles,
            })
    return out


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Self-test: du lieu tong hop dung tung mo hinh cu the (kiem tra nhan
    # dien dung hinh dang + dung boi canh xu huong).
    # ------------------------------------------------------------------
    def make_downtrend(n=15, start=100.0, step=1.0):
        closes = [start - i * step for i in range(n)]
        rows = []
        for cl in closes:
            rows.append({"open": cl + 0.3, "high": cl + 0.5, "low": cl - 0.5, "close": cl})
        return rows

    def make_uptrend(n=15, start=100.0, step=1.0):
        closes = [start + i * step for i in range(n)]
        rows = []
        for cl in closes:
            rows.append({"open": cl - 0.3, "high": cl + 0.5, "low": cl - 0.5, "close": cl})
        return rows

    def to_df(rows):
        idx = pd.bdate_range("2024-01-01", periods=len(rows))
        return pd.DataFrame(rows, index=idx[: len(rows)])

    # Test 1: Hammer sau downtrend (than nho gan dinh, bong duoi dai).
    # body_pct phai > 0.10 (khong bi bat lam Doji) nhung <= 0.35 (nho).
    rows = make_downtrend(12, start=100.0, step=1.0)
    last_close = rows[-1]["close"]
    rows.append({"open": last_close - 0.5, "high": last_close + 0.1, "low": last_close - 4.0, "close": last_close})
    df1 = to_df(rows)
    hits1 = detect_patterns(df1)
    names1 = [h.name for h in hits1 if h.index == len(df1) - 1]
    assert any("Hammer" in n and "ngược" not in n for n in names1), f"Khong phat hien Hammer: {names1}"
    print("[OK] Test 1 - Hammer sau downtrend duoc nhan dien:", names1)

    # Test 2: Hanging Man sau uptrend (hinh dang giong Hammer, nguoc boi canh)
    rows = make_uptrend(12, start=100.0, step=1.0)
    last_close = rows[-1]["close"]
    rows.append({"open": last_close - 0.5, "high": last_close + 0.1, "low": last_close - 4.0, "close": last_close})
    df2 = to_df(rows)
    hits2 = detect_patterns(df2)
    names2 = [h.name for h in hits2 if h.index == len(df2) - 1]
    assert any("Hanging Man" in n for n in names2), f"Khong phat hien Hanging Man: {names2}"
    print("[OK] Test 2 - Hanging Man sau uptrend duoc nhan dien:", names2)

    # Test 3: Bullish Engulfing sau downtrend
    rows = make_downtrend(12, start=100.0, step=1.0)
    prev_close = rows[-1]["close"]
    rows[-1] = {"open": prev_close + 1.0, "high": prev_close + 1.1, "low": prev_close - 0.1, "close": prev_close}
    rows.append({"open": prev_close - 0.3, "high": prev_close + 1.5, "low": prev_close - 0.4, "close": prev_close + 1.3})
    df3 = to_df(rows)
    hits3 = detect_patterns(df3)
    names3 = [h.name for h in hits3 if h.index == len(df3) - 1]
    assert any("Bullish Engulfing" in n for n in names3), f"Khong phat hien Bullish Engulfing: {names3}"
    print("[OK] Test 3 - Bullish Engulfing sau downtrend duoc nhan dien:", names3)

    # Test 4: Doji (than rat nho)
    rows = make_uptrend(10, start=100.0, step=0.8)
    last_close = rows[-1]["close"]
    rows.append({"open": last_close, "high": last_close + 1.0, "low": last_close - 1.0, "close": last_close + 0.02})
    df4 = to_df(rows)
    hits4 = detect_patterns(df4)
    names4 = [h.name for h in hits4 if h.index == len(df4) - 1]
    assert any("Doji" in n for n in names4), f"Khong phat hien Doji: {names4}"
    print("[OK] Test 4 - Doji duoc nhan dien:", names4)

    # Test 5: Three White Soldiers
    rows = make_downtrend(8, start=100.0, step=1.0)
    base = rows[-1]["close"]
    # moi nen mo trong khoang than nen truoc (o_next = o_prev + 0.9, tuc giua
    # than 1.8 diem cua nen truoc) roi dong cua cao hon - dung dinh nghia
    # Three White Soldiers (khong gap qua manh giua cac nen).
    o_cur = base + 0.2
    for k in range(3):
        cl = o_cur + 1.8
        rows.append({"open": o_cur, "high": cl + 0.2, "low": o_cur - 0.2, "close": cl})
        o_cur = o_cur + 0.9
    df5 = to_df(rows)
    hits5 = detect_patterns(df5)
    names5 = [h.name for h in hits5 if h.index == len(df5) - 1]
    assert any("Three White Soldiers" in n for n in names5), f"Khong phat hien Three White Soldiers: {names5}"
    print("[OK] Test 5 - Three White Soldiers duoc nhan dien:", names5)

    # Test 6: Morning Star (can du lieu truoc do >= lookback_trend+1 de
    # _prior_trend co the danh gia downtrend, khong bi roi ve "sideways" vi
    # thieu du lieu)
    rows = make_downtrend(15, start=100.0, step=1.0)
    c0 = rows[-1]["close"]
    rows[-1] = {"open": c0 + 2.0, "high": c0 + 2.1, "low": c0 - 0.1, "close": c0}
    rows.append({"open": c0 - 0.5, "high": c0 - 0.3, "low": c0 - 0.9, "close": c0 - 0.6})
    rows.append({"open": c0 - 0.4, "high": c0 + 1.7, "low": c0 - 0.5, "close": c0 + 1.6})
    df6 = to_df(rows)
    hits6 = detect_patterns(df6)
    names6 = [h.name for h in hits6 if h.index == len(df6) - 1]
    assert any("Morning Star" in n for n in names6), f"Khong phat hien Morning Star: {names6}"
    print("[OK] Test 6 - Morning Star duoc nhan dien:", names6)

    print("\nALL TESTS PASSED")
