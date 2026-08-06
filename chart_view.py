"""
chart_view.py
--------------
Phase 3 - module ve chart, TACH BIET voi UI (main.py) va voi phan tich
(analysis_engine.py). Chi nhan mot Axes (matplotlib) co san va ve len do -
main.py chi can nhung 1 FigureCanvas vao layout va goi plot_analysis().

Buoc 4 yeu cau ro: duong du bao KHONG duoc la 1 duong thang noi diem hien
tai voi diem sau 20 phien, ma phai la NHIEU DOAN (moi phien 1 gia tri rieng).
Vi cac forecast 3/10/20 phien trong analysis_engine deu la PREFIX cua CUNG
1 duong du bao (expected/lower/upper duoc tinh 1 lan cho max_horizon roi cat
ngan - xem generate_assessment), o day chi can ve DUY NHAT duong dai nhat
(thuong la 20 phien) la du hien thi day du ca 3 moc 3/10/20 tren cung 1 net.

Phase 4 (2026-07-29, theo yeu cau nguoi dung "ve mo hinh nen dang trung vao
bieu do"): them ve nen Nhat (candlestick) THAT cho vung lookback gan day
(thay cho duong close don gian truoc do), va chu thich cac mo hinh nen da
nhan dien duoc (candlestick_patterns.py) ngay tai vi tri xuat hien - dac biet
uu tien cac mo hinh GAN "hien tai" nhat vi day la thong tin nguoi dung can
xem "dang trung" luc phan tich. Ve nen thu cong bang Rectangle/Line2D (KHONG
them thu vien ngoai nhu mplfinance) de giu dependency gon nhu cac phase truoc.
"""
from __future__ import annotations

import numpy as np
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import analysis_engine as ae
import candlestick_patterns as cp

BULLISH_COLOR = "#2e7d32"
BEARISH_COLOR = "#c62828"
NEUTRAL_COLOR = "#9e9e9e"
ICHIMOKU_CLOUD_BULL = "#66bb6a"
ICHIMOKU_CLOUD_BEAR = "#ef5350"

# ten day trong candlestick_patterns.py hoi dai (co ca phan tieng Anh trong
# ngoac) - rut gon de chu thich tren chart khong bi roi.
_SHORT_NAME = {
    "Doji": "Doji",
    "Hammer": "Hammer",
    "Hanging Man": "Hanging Man",
    "Inverted Hammer": "Inv.Hammer",
    "Shooting Star": "Shooting Star",
    "Bullish Engulfing": "Bull.Engulf",
    "Bearish Engulfing": "Bear.Engulf",
    "Bullish Harami": "Bull.Harami",
    "Bearish Harami": "Bear.Harami",
    "Piercing Line": "Piercing",
    "Dark Cloud Cover": "Dark Cloud",
    "Morning Star": "Morning Star",
    "Evening Star": "Evening Star",
    "Three White Soldiers": "3 White Sold.",
    "Three Black Crows": "3 Black Crows",
}


def _short_pattern_name(full_name: str) -> str:
    for key, short in _SHORT_NAME.items():
        if full_name.startswith(key):
            return short
    return full_name.split(" (")[0]


def _draw_candles(ax, ohlc: pd.DataFrame, width_days: float = 0.6) -> None:
    """Ve nen Nhat (OHLC) thu cong: than nen = Rectangle, bong tren/duoi =
    Line2D. Khong dung mplfinance/them dependency - chi matplotlib co san
    (giong cach lam cua chart_view tu truoc)."""
    x_nums = mdates.date2num(ohlc.index.to_pydatetime())
    for xi, (_, row) in zip(x_nums, ohlc.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = BULLISH_COLOR if c >= o else BEARISH_COLOR
        ax.add_line(Line2D([xi, xi], [l, h], color=color, linewidth=0.8, zorder=2))
        body_low, body_high = min(o, c), max(o, c)
        # doji/gia dung yen co than = 0 -> ve 1 lat mong de van thay duoc nen
        height = max(body_high - body_low, (h - l) * 0.02 if h > l else 0.01)
        rect = Rectangle(
            (xi - width_days / 2, body_low), width_days, height,
            facecolor=color, edgecolor=color, linewidth=0.5, alpha=0.85, zorder=3,
        )
        ax.add_patch(rect)


def _annotate_patterns(ax, ohlc: pd.DataFrame, n_recent: int = 20) -> None:
    """Chu thich cac mo hinh nen nhan dien duoc TRONG vung du lieu dang ve
    (ohlc = phan da cat theo lookback_days), uu tien cac mo hinh gan 'hien
    tai' nhat (n_recent nen cuoi cua CHINH vung dang ve) - dung nhu yeu cau
    've mo hinh nen dang trung vao bieu do'."""
    hits = cp.recent_patterns(ohlc, n_recent=min(n_recent, len(ohlc)))
    if not hits:
        return

    # gom theo tung nen de khong chong chu thich neu 1 nen co nhieu mo hinh
    by_index: dict[int, list[dict]] = {}
    for h in hits:
        by_index.setdefault(h["index"], []).append(h)

    for idx, hits_at_idx in by_index.items():
        row = ohlc.iloc[idx]
        xi = mdates.date2num(ohlc.index[idx].to_pydatetime())
        names = [_short_pattern_name(h["name"]) for h in hits_at_idx]
        directions = {h["direction"] for h in hits_at_idx}
        if directions == {"bullish"}:
            color, marker, y, offset = BULLISH_COLOR, "^", row["low"], -14
        elif directions == {"bearish"}:
            color, marker, y, offset = BEARISH_COLOR, "v", row["high"], 14
        else:
            color, marker, y, offset = NEUTRAL_COLOR, "o", row["high"], 14

        ax.plot(xi, y, marker=marker, color=color, markersize=6, zorder=4)
        ax.annotate(
            "\n".join(names),
            xy=(xi, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            color=color,
            zorder=5,
        )


def _draw_ichimoku_cloud(ax, dates, senkou_a: np.ndarray, senkou_b: np.ndarray, alpha: float = 0.15,
                          label_bull: str | None = None, label_bear: str | None = None) -> None:
    """Ve may Ichimoku (Kumo) = vung giua Senkou Span A/B, doi mau theo chieu
    (xanh khi A>=B - "may tang", do khi A<B - "may giam") - dac trung hinh
    anh quan trong nhat cua Ichimoku, giup nhin nhanh xu huong trung han ma
    khong can doc tung chi so. Dung 2 lan fill_between voi mask `where=` de
    to 2 mau khac nhau tren cung 1 truc - NaN (chua du du lieu warm-up) tu
    dong bi bo qua vi so sanh voi NaN luon False."""
    senkou_a = np.asarray(senkou_a, dtype=float)
    senkou_b = np.asarray(senkou_b, dtype=float)
    bull_mask = senkou_a >= senkou_b
    bear_mask = senkou_a < senkou_b
    ax.fill_between(dates, senkou_a, senkou_b, where=bull_mask, color=ICHIMOKU_CLOUD_BULL,
                     alpha=alpha, interpolate=True, linewidth=0, label=label_bull)
    ax.fill_between(dates, senkou_a, senkou_b, where=bear_mask, color=ICHIMOKU_CLOUD_BEAR,
                     alpha=alpha, interpolate=True, linewidth=0, label=label_bear)


def plot_analysis(ax, symbol: str, history_df: pd.DataFrame, result, lookback_days: int = 120,
                   candle_lookback_days: int = 40) -> None:
    """Ve nen Nhat (vung gan day, `candle_lookback_days`) + EMA20/EMA50 (ca
    vung `lookback_days` de nhin xu huong dai hon) + duong du bao nhieu doan
    (co dai tin cay) + cac moc 3/10/20 phien + mui ten cuoi duong + chu thich
    mo hinh nen gan day.

    ax: matplotlib Axes (da duoc tao san boi noi goi, thuong la FigureCanvas
        trong PyQt). history_df: DataFrame OHLCV (index=date) cua chinh ma
        do, lay tu data_manager.get_price_df. result: AssessmentResult tra
        ve tu analysis_engine.generate_assessment.
    """
    ax.clear()

    hist = history_df.tail(lookback_days)
    candle_hist = history_df.tail(candle_lookback_days)

    ema20 = history_df["close"].ewm(span=20, adjust=False).mean().tail(lookback_days)
    ema50 = history_df["close"].ewm(span=50, adjust=False).mean().tail(lookback_days)

    # --- May Ichimoku (Kumo) + Tenkan/Kijun (2026-07-29, theo yeu cau nguoi
    # dung "bo sung Ichimoku") - ve TRUOC candlestick de may nam o nen, nen
    # khong che khuat nen/EMA/duong du bao ve sau. Tinh lai qua
    # analysis_engine.compute_indicators() de dung 1 nguon cong thuc duy
    # nhat voi noi da tinh cho classic_ta.py, tranh 2 noi tinh Ichimoku khac
    # nhau roi lech ket qua. ---
    ind_full = ae.compute_indicators(history_df)
    ind = ind_full.tail(lookback_days)
    if ind["ichimoku_senkou_a"].notna().any():
        _draw_ichimoku_cloud(
            ax, ind.index, ind["ichimoku_senkou_a"].to_numpy(), ind["ichimoku_senkou_b"].to_numpy(),
            label_bull="Mây Ichimoku (tăng)", label_bear="Mây Ichimoku (giảm)",
        )
        ax.plot(ind.index, ind["ichimoku_tenkan"], color="#e91e8c", linewidth=0.7, alpha=0.7, label="Tenkan-sen")
        ax.plot(ind.index, ind["ichimoku_kijun"], color="#3949ab", linewidth=0.9, alpha=0.8, label="Kijun-sen")

        # chieu may ve 26 phien tuong lai (dac trung hinh anh noi bat cua
        # Ichimoku) - dung ban THO (chua dich, "_fwd") cua 26 dong cuoi, vi
        # ban da dich (ichimoku_senkou_a/b) chi phan anh qua khu.
        last_date = history_df.index[-1]
        future_cloud_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=26)
        raw_a = ind_full["ichimoku_senkou_a_fwd"].tail(26).to_numpy()
        raw_b = ind_full["ichimoku_senkou_b_fwd"].tail(26).to_numpy()
        if not (np.isnan(raw_a).all() or np.isnan(raw_b).all()):
            _draw_ichimoku_cloud(ax, future_cloud_dates, raw_a, raw_b, alpha=0.10)

    ax.plot(hist.index, ema20, color="#7aa6c2", linewidth=0.9, alpha=0.8, label="EMA20")
    ax.plot(hist.index, ema50, color="#c2937a", linewidth=0.9, alpha=0.8, label="EMA50")

    _draw_candles(ax, candle_hist)
    # chi chu thich cac mo hinh THUC SU gan "hien tai" (vd 10 nen cuoi) de
    # khop voi yeu cau "ve mo hinh nen dang trung" va tranh chu thich chong
    # chat neu vung candle_lookback_days ve nhieu bien dong.
    _annotate_patterns(ax, candle_hist, n_recent=10)
    # legend gia thu cong cho nen (Rectangle khong tu sinh legend dep) - dung
    # 2 proxy artist cho tang/giam.
    candle_legend = [
        Line2D([0], [0], color=BULLISH_COLOR, lw=6, label="Nến tăng"),
        Line2D([0], [0], color=BEARISH_COLOR, lw=6, label="Nến giảm"),
    ]

    max_h = max(result.forecasts.keys())
    f = result.forecasts[max_h]

    as_of = pd.Timestamp(result.as_of_date)
    future_dates = pd.bdate_range(as_of + pd.Timedelta(days=1), periods=max_h)

    exp_prices = [result.current_price] + list(f["price_path"])
    low_prices = [result.current_price] + list(f["lower_price_path"])
    up_prices = [result.current_price] + list(f["upper_price_path"])
    x_dates = [as_of] + list(future_dates)

    # duong du bao: NHIEU DOAN (1 diem/phien), khong phai noi thang dau-cuoi
    ax.plot(
        x_dates, exp_prices, color="#d9534f", linewidth=1.6,
        marker="o", markersize=3, label=f"Dự báo kỳ vọng ({max_h} phiên)",
    )
    ax.fill_between(x_dates, low_prices, up_prices, color="#d9534f", alpha=0.15,
                     label="Dải tham khảo (phân vị 25-75 của hàng xóm lịch sử)")

    for h in sorted(result.forecasts.keys()):
        ret_pct = result.forecasts[h]["expected_return_pct"]
        ax.axvline(x_dates[h], color="#999999", linestyle="--", linewidth=0.7)
        ax.annotate(
            f"{h} phiên\n{ret_pct:+.1f}%",
            xy=(x_dates[h], exp_prices[h]),
            xytext=(0, 14 if ret_pct >= 0 else -26),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#d9534f",
        )

    # mui ten o doan cuoi cung, chi de nhan manh HUONG du bao - phan con lai
    # cua duong van la nhieu doan rieng nhu ve o tren, khong thay doi bang mui ten nay.
    if len(x_dates) >= 2:
        ax.annotate(
            "",
            xy=(x_dates[-1], exp_prices[-1]),
            xytext=(x_dates[-2], exp_prices[-2]),
            arrowprops=dict(arrowstyle="-|>", color="#d9534f", lw=1.6),
        )

    ax.axvline(as_of, color="#333333", linewidth=0.8)
    ax.set_title(f"{symbol} - giá & dự báo {max_h} phiên tới (k hiệu chỉnh = {result.calibrated_k})")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=candle_legend + handles, loc="upper left", fontsize=7)
    ax.grid(alpha=0.25)
    try:
        ax.figure.autofmt_xdate()
    except Exception:
        pass
