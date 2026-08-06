"""
classic_ta.py
--------------
Bang diem PHAN TICH KY THUAT CO DIEN (dinh tinh) - ket hop EMA/RSI/MACD/ADX/
volume + mo hinh nen gan day (candlestick_patterns.py) thanh 1 "cach doc"
bang ngon ngu thong thuong.

CANH BAO QUAN TRONG VE Y NGHIA CUA MODULE NAY (doc truoc khi dung):
Day la lop danh gia THEO KINH NGHIEM/QUY UOC cua phan tich ky thuat co dien
(vi du: "RSI>70 la qua mua", "EMA20>EMA50>EMA200 la xu huong tang manh"...).
NHUNG QUY UOC NAY **KHONG duoc kiem chung thong ke (KHONG co OOT hit-rate,
KHONG co backtest)** tren du lieu cua ma dang xet - no chi la kinh nghiem
pho bien duoc nhieu nguoi dung, khong dam bao dung cho tung ma/tung thoi
diem cu the.

Day CHINH LA sai lam cac notebook cu (Quant_VN) da mac phai: trinh bay diem
so dinh tinh nhu the la ket qua "model" dang tin cay (xem memory
vn_stock_prior_attempts_findings). De KHONG lap lai loi do, moi ket qua tra
ve tu module nay PHAI duoc hien thi kem canh bao "CHUA KIEM CHUNG THONG KE",
va nen duoc xem la BO SUNG dinh tinh - phu hop de tham khao nhanh cac tin
hieu ky thuat pho bien - cho ket qua da kiem chung OOT trong analysis_engine.py
(k-NN historical analogue, co hit-rate/MAE/baseline that su), CHU KHONG THAY
THE no.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Signal:
    name: str          # ten tin hieu (vd "Xu huong EMA")
    direction: str      # 'bullish' | 'bearish' | 'neutral'
    detail: str         # mo ta ngan gon


@dataclass
class ClassicScorecard:
    signals: list[Signal]
    score: int              # tong diem: +1 moi tin hieu bullish, -1 moi tin hieu bearish
    max_score: int           # so tin hieu duoc cham (de tinh ty le)
    verdict: str             # "Tích cực" / "Tiêu cực" / "Trung lập"
    verdict_detail: str
    summary_text: str        # van ban tong hop, tieng Viet, kem canh bao
    disclaimer: str = field(default=(
        "⚠ Bảng điểm này là đánh giá ĐỊNH TÍNH theo kinh nghiệm phân tích kỹ "
        "thuật cổ điển (EMA/RSI/MACD/ADX/volume/mô hình nến) - CHƯA được kiểm "
        "chứng thống kê (không có hit-rate/backtest ngoài mẫu) trên chính mã "
        "này. Chỉ nên dùng để tham khảo nhanh các tín hiệu kỹ thuật phổ biến, "
        "KHÔNG thay thế phần dự báo đã kiểm chứng OOT ở trên."
    ))


def _score_trend(row: pd.Series) -> Signal:
    ema20, ema50, ema200 = row.get("ema20"), row.get("ema50"), row.get("ema200")
    if any(pd.isna(x) for x in (ema20, ema50, ema200)):
        return Signal("Xu hướng EMA", "neutral", "không đủ dữ liệu EMA200")
    if ema20 > ema50 > ema200:
        return Signal("Xu hướng EMA", "bullish", "EMA20 > EMA50 > EMA200 - xếp lớp tăng chuẩn (uptrend rõ)")
    if ema20 < ema50 < ema200:
        return Signal("Xu hướng EMA", "bearish", "EMA20 < EMA50 < EMA200 - xếp lớp giảm chuẩn (downtrend rõ)")
    if ema20 > ema50:
        return Signal("Xu hướng EMA", "bullish", "EMA20 > EMA50 - xu hướng ngắn hạn nghiêng tăng, chưa xếp lớp hoàn chỉnh")
    if ema20 < ema50:
        return Signal("Xu hướng EMA", "bearish", "EMA20 < EMA50 - xu hướng ngắn hạn nghiêng giảm, chưa xếp lớp hoàn chỉnh")
    return Signal("Xu hướng EMA", "neutral", "các đường EMA đan xen, chưa rõ xu hướng")


def _score_rsi(row: pd.Series) -> Signal:
    rsi = row.get("rsi14")
    if pd.isna(rsi):
        return Signal("RSI(14)", "neutral", "không đủ dữ liệu")
    if rsi >= 70:
        return Signal("RSI(14)", "bearish", f"RSI={rsi:.0f} ở vùng quá mua - rủi ro điều chỉnh ngắn hạn")
    if rsi <= 30:
        return Signal("RSI(14)", "bullish", f"RSI={rsi:.0f} ở vùng quá bán - có thể hồi phục ngắn hạn")
    if rsi >= 55:
        return Signal("RSI(14)", "bullish", f"RSI={rsi:.0f} - động lượng nghiêng tăng")
    if rsi <= 45:
        return Signal("RSI(14)", "bearish", f"RSI={rsi:.0f} - động lượng nghiêng giảm")
    return Signal("RSI(14)", "neutral", f"RSI={rsi:.0f} - trung tính")


def _score_macd(row: pd.Series) -> Signal:
    hist = row.get("macd_hist")
    macd = row.get("macd")
    signal = row.get("macd_signal")
    if any(pd.isna(x) for x in (hist, macd, signal)):
        return Signal("MACD", "neutral", "không đủ dữ liệu")
    if macd > signal and hist > 0:
        return Signal("MACD", "bullish", "đường MACD trên đường tín hiệu, histogram dương - động lượng tăng")
    if macd < signal and hist < 0:
        return Signal("MACD", "bearish", "đường MACD dưới đường tín hiệu, histogram âm - động lượng giảm")
    return Signal("MACD", "neutral", "MACD/tín hiệu đan xen - động lượng chưa rõ ràng")


def _score_adx(row: pd.Series) -> Signal:
    adx = row.get("adx14")
    trend_score = row.get("trend_score")
    if pd.isna(adx):
        return Signal("ADX(14) - sức mạnh xu hướng", "neutral", "không đủ dữ liệu")
    if adx < 20:
        return Signal("ADX(14) - sức mạnh xu hướng", "neutral", f"ADX={adx:.0f} - xu hướng yếu/đi ngang, tín hiệu xu hướng khác nên giảm trọng số")
    # ADX cao chi xac nhan xu huong hien co MANH, khong tu no cho biet huong
    direction = "bullish" if (trend_score is not None and not pd.isna(trend_score) and trend_score > 0) else \
                ("bearish" if (trend_score is not None and not pd.isna(trend_score) and trend_score < 0) else "neutral")
    return Signal("ADX(14) - sức mạnh xu hướng", direction, f"ADX={adx:.0f} - xu hướng hiện tại khá mạnh, củng cố chiều của các tín hiệu khác")


def _score_volume(row: pd.Series) -> Signal:
    vol_ratio = row.get("vol_ratio20")
    trend_score = row.get("trend_score")
    if pd.isna(vol_ratio):
        return Signal("Khối lượng", "neutral", "không đủ dữ liệu")
    if vol_ratio >= 1.3:
        direction = "bullish" if (trend_score is not None and not pd.isna(trend_score) and trend_score > 0) else \
                    ("bearish" if (trend_score is not None and not pd.isna(trend_score) and trend_score < 0) else "neutral")
        return Signal("Khối lượng", direction, f"Khối lượng gấp {vol_ratio:.1f}x trung bình 20 phiên - xác nhận lực của xu hướng giá hiện tại")
    if vol_ratio <= 0.6:
        return Signal("Khối lượng", "neutral", f"Khối lượng chỉ {vol_ratio:.1f}x trung bình 20 phiên - dòng tiền yếu, tín hiệu giá kém tin cậy hơn")
    return Signal("Khối lượng", "neutral", f"Khối lượng ở mức bình thường ({vol_ratio:.1f}x trung bình 20 phiên)")


def _score_ichimoku_kumo(row: pd.Series) -> Signal:
    """Gia so voi may Ichimoku (Kumo, vung giua Senkou Span A/B) - 1 trong
    nhung tin hieu Ichimoku duoc dung pho bien nhat: gia tren may = ho tro
    (bullish), gia duoi may = khang cu (bearish), gia trong may = di ngang/
    chua ro (neutral). Ket hop them "xac nhan Chikou" (gia hien tai so voi
    gia 26 phien truoc) lam chi tiet bo sung, khong tach thanh tin hieu rieng
    de tranh Ichimoku chiem qua nhieu trong so trong bang diem tong."""
    close = row.get("close")
    senkou_a = row.get("ichimoku_senkou_a")
    senkou_b = row.get("ichimoku_senkou_b")
    if any(pd.isna(x) for x in (close, senkou_a, senkou_b)):
        return Signal("Mây Ichimoku (Kumo)", "neutral", "không đủ dữ liệu (cần ~78 phiên)")

    cloud_top, cloud_bottom = max(senkou_a, senkou_b), min(senkou_a, senkou_b)
    chikou_ref = row.get("ichimoku_chikou_ref")
    chikou_txt = ""
    if chikou_ref is not None and not pd.isna(chikou_ref):
        chikou_txt = "; Chikou xác nhận tăng" if close > chikou_ref else "; Chikou xác nhận giảm"

    if close > cloud_top:
        return Signal("Mây Ichimoku (Kumo)", "bullish", f"Giá trên mây - mây đóng vai trò hỗ trợ{chikou_txt}")
    if close < cloud_bottom:
        return Signal("Mây Ichimoku (Kumo)", "bearish", f"Giá dưới mây - mây đóng vai trò kháng cự{chikou_txt}")
    return Signal("Mây Ichimoku (Kumo)", "neutral", f"Giá đang nằm TRONG mây - vùng giằng co/chưa rõ xu hướng{chikou_txt}")


def _score_ichimoku_cross(row: pd.Series) -> Signal:
    """Tenkan-sen cat Kijun-sen - tuong tu 1 duong trung binh nhanh cat
    duong trung binh cham (nhu MA cross), thuong dung de bat tin hieu doi
    chieu ngan-trung han som hon EMA20/EMA50 thong thuong."""
    tenkan, kijun = row.get("ichimoku_tenkan"), row.get("ichimoku_kijun")
    if any(pd.isna(x) for x in (tenkan, kijun)):
        return Signal("Tenkan/Kijun", "neutral", "không đủ dữ liệu")
    if tenkan > kijun:
        return Signal("Tenkan/Kijun", "bullish", f"Tenkan ({tenkan:.1f}) > Kijun ({kijun:.1f}) - động lượng ngắn hạn nghiêng tăng")
    if tenkan < kijun:
        return Signal("Tenkan/Kijun", "bearish", f"Tenkan ({tenkan:.1f}) < Kijun ({kijun:.1f}) - động lượng ngắn hạn nghiêng giảm")
    return Signal("Tenkan/Kijun", "neutral", "Tenkan ≈ Kijun - chưa rõ chiều")


def _score_candles(candle_hits: list[dict]) -> list[Signal]:
    signals = []
    for hit in candle_hits:
        date_str = hit["date"].strftime("%Y-%m-%d") if hasattr(hit["date"], "strftime") else str(hit["date"])
        direction = hit["direction"] if hit["direction"] in ("bullish", "bearish") else "neutral"
        signals.append(Signal(
            f"Mô hình nến: {hit['name']}",
            direction,
            f"xuất hiện ngày {date_str}",
        ))
    return signals


def build_classic_scorecard(latest_row: pd.Series, candle_hits: list[dict] | None = None) -> ClassicScorecard:
    """Xay dung bang diem dinh tinh tu 1 dong chi bao moi nhat (tu
    analysis_engine.compute_indicators/compute_market_context) va danh sach
    mo hinh nen gan day (tu candlestick_patterns.recent_patterns). Chi dung
    de tham khao nhanh - xem canh bao trong docstring module."""
    candle_hits = candle_hits or []

    signals = [
        _score_trend(latest_row),
        _score_rsi(latest_row),
        _score_macd(latest_row),
        _score_adx(latest_row),
        _score_volume(latest_row),
        _score_ichimoku_kumo(latest_row),
        _score_ichimoku_cross(latest_row),
    ]
    signals.extend(_score_candles(candle_hits))

    scored = [s for s in signals if s.direction in ("bullish", "bearish")]
    score = sum(1 if s.direction == "bullish" else -1 for s in scored)
    max_score = len(scored)

    if max_score == 0:
        verdict, verdict_detail = "Trung lập", "không đủ tín hiệu rõ ràng để đánh giá"
    else:
        ratio = score / max_score
        if ratio >= 0.4:
            verdict, verdict_detail = "Tích cực", f"{sum(1 for s in scored if s.direction=='bullish')}/{max_score} tín hiệu nghiêng tăng"
        elif ratio <= -0.4:
            verdict, verdict_detail = "Tiêu cực", f"{sum(1 for s in scored if s.direction=='bearish')}/{max_score} tín hiệu nghiêng giảm"
        else:
            verdict, verdict_detail = "Trung lập/hỗn hợp", f"tín hiệu tăng và giảm gần cân bằng ({max_score} tín hiệu được chấm)"

    lines = [f"Đánh giá kỹ thuật cổ điển (định tính): {verdict} - {verdict_detail}."]
    for s in signals:
        arrow = {"bullish": "▲", "bearish": "▼", "neutral": "•"}[s.direction]
        lines.append(f"  {arrow} {s.name}: {s.detail}")

    scorecard = ClassicScorecard(
        signals=signals,
        score=score,
        max_score=max_score,
        verdict=verdict,
        verdict_detail=verdict_detail,
        summary_text="\n".join(lines),
    )
    return scorecard


if __name__ == "__main__":
    import numpy as np

    # Test 1: tat ca tin hieu bullish ro rang (bao gom Ichimoku: gia tren may,
    # Tenkan>Kijun, Chikou > gia 26 phien truoc)
    row_bull = pd.Series({
        "ema20": 105, "ema50": 100, "ema200": 90,
        "rsi14": 62, "macd": 1.2, "macd_signal": 0.8, "macd_hist": 0.4,
        "adx14": 30, "trend_score": 1.0, "vol_ratio20": 1.6,
        "close": 110, "ichimoku_tenkan": 106, "ichimoku_kijun": 102,
        "ichimoku_senkou_a": 100, "ichimoku_senkou_b": 95, "ichimoku_chikou_ref": 95,
    })
    sc = build_classic_scorecard(row_bull, candle_hits=[
        {"date": pd.Timestamp("2026-07-28"), "name": "Bullish Engulfing", "direction": "bullish"},
    ])
    assert sc.verdict == "Tích cực", sc.verdict
    assert sc.score > 0
    print("[OK] Test 1 - kich ban toan bullish -> verdict Tich cuc:", sc.verdict, sc.verdict_detail)

    # Test 2: tat ca tin hieu bearish ro rang (Ichimoku: gia duoi may, Tenkan<Kijun)
    row_bear = pd.Series({
        "ema20": 90, "ema50": 100, "ema200": 110,
        "rsi14": 35, "macd": -1.0, "macd_signal": -0.6, "macd_hist": -0.4,
        "adx14": 28, "trend_score": -1.0, "vol_ratio20": 1.5,
        "close": 85, "ichimoku_tenkan": 92, "ichimoku_kijun": 98,
        "ichimoku_senkou_a": 100, "ichimoku_senkou_b": 105, "ichimoku_chikou_ref": 95,
    })
    sc2 = build_classic_scorecard(row_bear, candle_hits=[
        {"date": pd.Timestamp("2026-07-28"), "name": "Bearish Engulfing", "direction": "bearish"},
    ])
    assert sc2.verdict == "Tiêu cực", sc2.verdict
    assert sc2.score < 0
    print("[OK] Test 2 - kich ban toan bearish -> verdict Tieu cuc:", sc2.verdict, sc2.verdict_detail)

    # Test 3: tin hieu hon hop / trung lap (Ichimoku: gia nam TRONG may)
    row_mixed = pd.Series({
        "ema20": 100.5, "ema50": 100, "ema200": 100,
        "rsi14": 50, "macd": 0.05, "macd_signal": 0.06, "macd_hist": -0.01,
        "adx14": 15, "trend_score": 0.05, "vol_ratio20": 0.9,
        "close": 100, "ichimoku_tenkan": 100, "ichimoku_kijun": 100,
        "ichimoku_senkou_a": 102, "ichimoku_senkou_b": 98, "ichimoku_chikou_ref": 100,
    })
    sc3 = build_classic_scorecard(row_mixed)
    assert "Trung lập" in sc3.verdict or sc3.max_score == 0, sc3.verdict
    print("[OK] Test 3 - kich ban trung lap:", sc3.verdict, sc3.verdict_detail)

    # Test 4: disclaimer luon co mat trong scorecard
    assert "CHƯA được kiểm" in sc.disclaimer
    print("[OK] Test 4 - disclaimer 'chua kiem chung thong ke' co mat")

    # Test 5: tin hieu Ichimoku rieng - kiem tra dung 3 truong hop gia so voi may
    kumo_bull = _score_ichimoku_kumo(pd.Series({
        "close": 110, "ichimoku_senkou_a": 100, "ichimoku_senkou_b": 95, "ichimoku_chikou_ref": 90,
    }))
    assert kumo_bull.direction == "bullish", kumo_bull
    kumo_bear = _score_ichimoku_kumo(pd.Series({
        "close": 85, "ichimoku_senkou_a": 100, "ichimoku_senkou_b": 95, "ichimoku_chikou_ref": 90,
    }))
    assert kumo_bear.direction == "bearish", kumo_bear
    kumo_neutral = _score_ichimoku_kumo(pd.Series({
        "close": 97, "ichimoku_senkou_a": 100, "ichimoku_senkou_b": 95, "ichimoku_chikou_ref": 90,
    }))
    assert kumo_neutral.direction == "neutral", kumo_neutral
    cross_bull = _score_ichimoku_cross(pd.Series({"ichimoku_tenkan": 105, "ichimoku_kijun": 100}))
    assert cross_bull.direction == "bullish", cross_bull
    cross_bear = _score_ichimoku_cross(pd.Series({"ichimoku_tenkan": 95, "ichimoku_kijun": 100}))
    assert cross_bear.direction == "bearish", cross_bear
    print("[OK] Test 5 - tin hieu Ichimoku (Kumo trên/dưới/trong mây, Tenkan/Kijun cross) đúng chiều")

    print("\nRationale mau (kich ban bullish):\n" + sc.summary_text)
    print("\nALL TESTS PASSED")
