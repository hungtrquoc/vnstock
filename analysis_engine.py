"""
analysis_engine.py
-------------------
Phase 2 - module phan tich ky thuat rieng biet (KHONG dung chung code voi
data_manager.py). Day la "ham python rieng cho muc tieu phan tich" theo yeu
cau Buoc 3.

VI SAO KHONG DUNG LAI CACH CU (xem memory vn_stock_prior_attempts_findings):
Cac notebook cu train 1 model phan loai (XGBoost/LSTM...) rieng cho tung ma,
chi voi ~1800-2200 dong du lieu/ma. Sau khi sua het loi leakage, cv_score van
chi ~0.40-0.55 tren gan 90 ma - xap xi baseline theo lop da so, tuc KHONG co
edge thuc su. Nguyen nhan: qua it du lieu de fit 1 model phuc tap rieng cho
tung ma, va khong chia se thong tin giua cac ma.

PHUONG PHAP (v3, cap nhat 2026-07-29 theo yeu cau "cai tien do tin cay"):
1. Tinh 1 bo chi bao ky thuat + vi the tuong doi so voi VNINDEX cho tung ngay
   (trend, momentum, bien dong, volume, market regime) - xem compute_indicators/
   compute_market_context.
2. Du bao bang "historical analogue" (k-nearest-neighbor tren khong gian
   TRANG THAI ky thuat): tim trong LICH SU (chi trong QUA KHU nghiem ngat so
   voi ngay dang xet) cac giai doan co trang thai tuong tu nhat, lay trung
   binh + phan tan cua duong di gia THUC TE sau do lam du bao 3/10/20/50 phien
   toi - ket qua la 1 DUONG NHIEU DOAN, khong phai 1 duong thang.
3. QUAN TRONG - GOP HANG XOM THEO NHOM NGANH (moi, xem build_pool_source):
   thay vi chi tim "hang xom" trong lich su cua RIENG 1 ma (qua it du lieu de
   co ket qua dang tin cay thong ke, day la diem yeu goc re khien hit-rate
   thap/khong on dinh o v2), gio day pool hang xom GOM CA cac ma cung nhom
   nganh/beta (xem sector_map.py) - tang co mau dang ke, tang co hoi phat
   hien tin hieu that neu no ton tai. Neu khong co peer nao (ma le, chua co
   trong sector_map), tu dong lui ve che do 1 ma rieng le (an toan, khong loi).
4. QUAN TRONG - WALK-FORWARD NHIEU GIAI DOAN (moi, xem walk_forward_evaluate):
   thay vi chia 1 lan Train/Val/OOT co dinh (de bi may-rui neu split roi vao
   giai doan thi truong dac biet), gio day chia thanh nhieu fold lien tiep
   (moi fold: chon k tren VAL cua fold do, danh gia tren OOT cua fold do),
   roi GOP ket qua tat ca fold lai - vua tang co mau cho kiem dinh thong ke,
   vua cho biet ket qua co ON DINH qua nhieu giai doan hay chi la may-rui.
5. QUAN TRONG - KIEM DINH THONG KE (moi, xem _binomial_pvalue): thay vi chi
   so sanh hit-rate voi baseline bang 1 nguong tuy y (+3 diem % nhu ban dau),
   gio day dung kiem dinh nhi thuc (binomial test, xap xi chuan, khong can
   scipy) de tra loi cau hoi: "hit-rate quan sat duoc co THAT SU khac 50%
   (tuc co tin hieu huong di that), hay chi la nhieu ngau nhien voi co mau
   nay?". p-value < SIGNIFICANCE_ALPHA moi duoc coi la "co y nghia thong ke".
6. Danh gia TRUNG THUC tren OOT (hit rate huong di, MAE, p-value, do on dinh
   qua cac fold) TRUOC khi dua ra nhan dinh cho hien tai - day la "do tin cay
   thuc te" hien thi cho nguoi dung, khac voi cac notebook cu bao win rate ao
   do leakage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (3, 10, 20, 50)  # them 50 phien theo yeu cau nguoi dung 2026-08-03
DEFAULT_K_GRID = (10, 20, 30, 50)
WARMUP_ROWS = 210          # so dong dau bi bo vi chi bao (EMA200...) chua on dinh
MIN_USABLE_ROWS = 400      # can it nhat khoang nay de walk-forward co y nghia

N_FOLDS = 4                     # so fold walk-forward
INIT_TRAIN_FRAC = 0.4           # ty le du lieu dau danh rieng cho "train khoi dong", khong danh gia
FOLD_VAL_FRAC = 0.08            # ty le (tren tong so dong usable) danh cho VAL cua moi fold
MIN_FOLD_VAL_LEN = 30           # do dai toi thieu cua VAL moi fold (dong)
SIGNIFICANCE_ALPHA = 0.10       # nguong p-value de coi la "co y nghia thong ke"


# ---------------------------------------------------------------------------
# 1) Chi bao ky thuat
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(0.0)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tinh cac chi bao ky thuat chuan tren 1 DataFrame OHLCV (index = date).
    Khong sua doi df goc; tra ve DataFrame moi voi them cac cot chi bao."""
    out = df.copy()
    close = out["close"]

    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()

    out["rsi14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    tr = _true_range(out)
    out["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_upper"] = sma20 + 2 * std20
    out["bb_lower"] = sma20 - 2 * std20
    bb_width = (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
    out["bb_pctb"] = (close - out["bb_lower"]) / bb_width

    out["adx14"] = _adx(out, 14)

    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["vol_ratio20"] = out["volume"] / out["vol_sma20"].replace(0, np.nan)

    # bat thuong khoi luong tren nen dai hon (60 phien) - nhay hon voi dot bien
    # khoi luong RIENG cua ma nay (dau hieu dong tien vao/ra dac thu tung co
    # phieu), thay vi chi so sanh voi 20 phien gan nhat.
    vol_mean60 = out["volume"].rolling(60).mean()
    vol_std60 = out["volume"].rolling(60).std()
    out["vol_zscore60"] = (out["volume"] - vol_mean60) / vol_std60.replace(0, np.nan)

    obv_dir = np.sign(close.diff()).fillna(0)
    out["obv"] = (obv_dir * out["volume"]).cumsum()

    # feature chuan hoa theo ATR de so sanh duoc giua cac ma khac gia
    out["trend_score"] = (close - out["ema50"]) / out["atr14"].replace(0, np.nan)
    out["macd_hist_norm"] = out["macd_hist"] / out["atr14"].replace(0, np.nan)

    # --- Ichimoku Kinko Hyo (2026-07-29, theo yeu cau nguoi dung) ---
    # Chi dung cho classic_ta.py (dinh tinh) va ve chart - KHONG dua vao
    # FEATURE_COLS/k-NN de tranh lam thay doi pipeline thong ke vua duoc
    # kiem chung ky (walk-forward + p-value + peer pooling). Neu sau nay
    # muon dua Ichimoku vao k-NN thi phai backtest lai tu dau, khong duoc
    # them "chui" vao feature dang dung.
    #
    # Chu ky chuan: Tenkan(9)/Kijun(26)/Senkou B(52), do dich (displacement)
    # = 26 phien. Senkou A/B duoc TINH tai ngay t nhung VE (hien thi) tai
    # ngay t+26 - nen "may" nhin thay tai ngay hien tai thuc ra duoc tinh tu
    # 26 phien TRUOC do -> can .shift(26) de dong bo dung ngay khi so sanh
    # gia hien tai voi may. Ban khong-dich (hau to "_fwd") duoc giu lai rieng
    # de ve phan "may" chieu ve tuong lai 26 phien tren chart (dac trung hinh
    # anh noi bat cua Ichimoku).
    high9 = out["high"].rolling(9).max()
    low9 = out["low"].rolling(9).min()
    out["ichimoku_tenkan"] = (high9 + low9) / 2

    high26 = out["high"].rolling(26).max()
    low26 = out["low"].rolling(26).min()
    out["ichimoku_kijun"] = (high26 + low26) / 2

    raw_senkou_a = (out["ichimoku_tenkan"] + out["ichimoku_kijun"]) / 2
    high52 = out["high"].rolling(52).max()
    low52 = out["low"].rolling(52).min()
    raw_senkou_b = (high52 + low52) / 2

    out["ichimoku_senkou_a"] = raw_senkou_a.shift(26)      # may "nhin thay" tai ngay hien tai
    out["ichimoku_senkou_b"] = raw_senkou_b.shift(26)
    out["ichimoku_senkou_a_fwd"] = raw_senkou_a            # ban tho, dung de chieu may ve tuong lai
    out["ichimoku_senkou_b_fwd"] = raw_senkou_b

    # Chikou (lagging span) = gia dong cua hien tai, ve lui 26 phien ve truoc.
    # De so sanh tin hieu chi voi 1 dong du lieu moi nhat (classic_ta.py),
    # luu san "gia dong 26 phien truoc" tai MOI dong hien tai - so sanh
    # close hien tai > cot nay <=> chikou dang o tren gia qua khu (xac nhan tang).
    out["ichimoku_chikou_ref"] = close.shift(26)

    return out


def compute_market_context(stock_ind: pd.DataFrame, vnindex_ind: pd.DataFrame) -> pd.DataFrame:
    """Ghep them cac cot the hien vi the RIENG CUA MA (idiosyncratic) sau khi
    da loai bo phan bien dong chung voi VNINDEX, va regime cua thi truong
    chung (dua tren chinh VNINDEX).

    QUAN TRONG (sua loi "du bao giong nhau giua cac ma"): ban dau dung
    rel_strength = stock_return - index_return (gia dinh beta=1 cho MOI ma).
    Voi cac ma co beta gan 1 (rat pho bien o nhom blue-chip VN), phan lon
    bien dong cua ma chi la phan anh cua VNINDEX -> vector trang thai cua
    nhieu ma o CUNG 1 ngay se gan giong nhau (vi VNINDEX la 1 chuoi duy nhat,
    dung chung cho tat ca), khien k-NN tim ra cac "hang xom" gan nhu giong
    nhau giua cac ma va du bao ra ket qua tuong tu nhau. Sua bang cach uoc
    luong beta cuon (rolling) rieng cho tung ma, roi tinh PHAN DU (residual)
    sau khi tru di phan bien dong da duoc beta du bao - day moi la tin hieu
    THUC SU dac thu cua rieng ma do, doc lap voi VNINDEX.
    """
    out = stock_ind.copy()

    daily_stock_ret = stock_ind["close"].pct_change()
    daily_idx_ret = vnindex_ind["close"].pct_change().reindex(out.index)

    roll_cov = daily_stock_ret.rolling(60).cov(daily_idx_ret)
    roll_var = daily_idx_ret.rolling(60).var()
    beta60 = (roll_cov / roll_var.replace(0, np.nan)).clip(-3, 3)
    out["beta60"] = beta60

    stock_ret20 = stock_ind["close"].pct_change(20)
    stock_ret60 = stock_ind["close"].pct_change(60)
    idx_ret20 = vnindex_ind["close"].pct_change(20).reindex(out.index)
    idx_ret60 = vnindex_ind["close"].pct_change(60).reindex(out.index)

    # phan du sau khi tru di phan "chi la theo thi truong" (beta * index_return)
    out["beta_resid_20"] = stock_ret20 - beta60 * idx_ret20
    out["beta_resid_60"] = stock_ret60 - beta60 * idx_ret60

    # vi tri gia trong kenh gia 120 phien cua CHINH MA nay (0=day 120 phien,
    # 1=dinh 120 phien) - dac thu rieng, khong phu thuoc VNINDEX
    roll_high120 = stock_ind["high"].rolling(120).max()
    roll_low120 = stock_ind["low"].rolling(120).min()
    out["range_pct_120"] = (stock_ind["close"] - roll_low120) / (roll_high120 - roll_low120).replace(0, np.nan)

    ema50_idx = vnindex_ind["ema50"].reindex(out.index)
    ema200_idx = vnindex_ind["ema200"].reindex(out.index)
    ema50_slope = ema50_idx.diff(10)

    regime = pd.Series("sideways", index=out.index)
    regime[(ema50_idx > ema200_idx) & (ema50_slope > 0)] = "uptrend"
    regime[(ema50_idx < ema200_idx) & (ema50_slope < 0)] = "downtrend"
    out["market_regime"] = regime

    return out


# ---------------------------------------------------------------------------
# 2) Xay dung bang feature "dai" (long-format) - dung chung cho target va pool
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "trend_score",
    "rsi14",
    "macd_hist_norm",
    "bb_pctb",
    "vol_ratio20",
    "vol_zscore60",
    "adx14",
    "beta_resid_20",   # idiosyncratic - da loai bo phan chi la theo VNINDEX
    "beta_resid_60",   # idiosyncratic - da loai bo phan chi la theo VNINDEX
    "range_pct_120",   # vi tri gia trong kenh 120 phien cua chinh ma
]


def build_feature_table(symbol: str, stock_df: pd.DataFrame, vnindex_ind: pd.DataFrame) -> pd.DataFrame:
    """Tinh chi bao + boi canh thi truong cho 1 ma, tra ve bang 'dai'
    (long-format, 1 dong/ngay) voi cot 'date'/'symbol' + tat ca cot chi bao
    (bao gom ca FEATURE_COLS lan cac cot mo ta nhu beta60/market_regime -
    can giu lai de _describe_state dung duoc). Da bo warm-up va NaN.
    Dung lam khoi xay dung chung cho ca truong hop 1 ma rieng le va pool
    nhieu ma (xem build_pool_source)."""
    stock_ind = compute_indicators(stock_df)
    full = compute_market_context(stock_ind, vnindex_ind)
    full = full.dropna(subset=FEATURE_COLS + ["close"])
    full = full.iloc[WARMUP_ROWS:] if len(full) > WARMUP_ROWS else full
    full = full.reset_index().rename(columns={full.index.name or "index": "date"})
    full["symbol"] = symbol
    return full.reset_index(drop=True)


def _attach_forward_paths(table: pd.DataFrame, max_horizon: int) -> pd.DataFrame:
    """Them cot 'fwd_path' (mang % thay doi gia h=1..max_horizon phien sau)
    cho MOI dong, CHI GIU LAI cac dong co du du lieu tuong lai (bo
    max_horizon dong cuoi - khong co outcome that de danh gia/lam hang xom).
    Tinh RIENG cho tung ma (moi ma co chuoi ngay giao dich rieng, khong the
    tinh xuyen ma)."""
    if len(table) <= max_horizon:
        empty = table.iloc[0:0].copy()
        empty["fwd_path"] = pd.Series(dtype=object)
        return empty
    close = table["close"].to_numpy()
    usable = table.iloc[: len(table) - max_horizon].copy()
    paths = _forward_return_paths(close, np.arange(len(usable)), max_horizon)
    usable["fwd_path"] = list(paths)
    return usable.reset_index(drop=True)


def build_pool_source(
    symbol: str,
    stock_df: pd.DataFrame,
    vnindex_df: pd.DataFrame,
    peer_price_dfs: Optional[dict[str, pd.DataFrame]],
    max_horizon: int,
) -> dict:
    """Xay dung du lieu goc cho ca muc tieu (target) va pool hang xom (gom
    ca chinh no + cac ma cung nhom nganh/beta neu co - xem sector_map.py).

    Tra ve dict:
      - target_full: bang feature DAY DU cua target (bao gom dong cuoi/hom
        nay, CHUA co fwd_path vi dong cuoi chua co tuong lai) - dung de mo
        ta trang thai hien tai va lam query cho du bao "song".
      - target_usable: bang feature cua target CO fwd_path (da bo cac dong
        cuoi thieu du lieu tuong lai) - dung lam QUERY cho calibrate/OOT.
      - pool_source: TAT CA cac ma (target + peers) da co fwd_path, gop lai
        thanh 1 bang dai - dung lam nguon HANG XOM (loc theo NGAY, khong
        phai vi tri, trong walk-forward - xem walk_forward_evaluate).
      - peers_used: danh sach ma peer THUC SU dung duoc (co du du lieu).
    """
    vnindex_ind = compute_indicators(vnindex_df)

    target_full = build_feature_table(symbol, stock_df, vnindex_ind)
    target_usable = _attach_forward_paths(target_full, max_horizon)

    pool_frames = [target_usable]
    peers_used: list[str] = []
    for peer_symbol, peer_df in (peer_price_dfs or {}).items():
        if peer_symbol == symbol or peer_df is None or peer_df.empty:
            continue
        try:
            peer_full = build_feature_table(peer_symbol, peer_df, vnindex_ind)
            peer_usable = _attach_forward_paths(peer_full, max_horizon)
        except Exception:
            continue
        if len(peer_usable) < MIN_USABLE_ROWS // 2:
            continue  # qua it du lieu, bo qua de tranh nhieu chat luong thap
        pool_frames.append(peer_usable)
        peers_used.append(peer_symbol)

    pool_source = pd.concat(pool_frames, ignore_index=True) if pool_frames else target_usable

    return {
        "target_full": target_full,
        "target_usable": target_usable,
        "pool_source": pool_source,
        "peers_used": peers_used,
    }


# ---------------------------------------------------------------------------
# 3) k-NN forecast tren khong gian trang thai
# ---------------------------------------------------------------------------

def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(std == 0, 1.0, std)
    return (features - mean) / std_safe


def _forward_return_paths(close: np.ndarray, indices: np.ndarray, max_horizon: int) -> np.ndarray:
    """Voi moi index i trong `indices`, tra ve mang shape (len(indices), max_horizon)
    la % thay doi gia tu ngay i den ngay i+1..i+max_horizon. Gia dinh i+max_horizon
    luon nam trong pham vi du lieu (nguoi goi phai dam bao dieu nay)."""
    n = len(indices)
    paths = np.empty((n, max_horizon), dtype=float)
    for row, i in enumerate(indices):
        base = close[i]
        for h in range(1, max_horizon + 1):
            paths[row, h - 1] = close[i + h] / base - 1.0
    return paths


def _knn_forecast(
    query_feat: np.ndarray,
    pool_feat: np.ndarray,
    pool_fwd_paths: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tra ve (expected_path, lower_band, upper_band) - moi mang shape (max_horizon,).
    expected_path = trung binh duong di cua k hang xom gan nhat; band = phan vi
    25/75 cua cac hang xom do (the hien do phan tan / bat dinh, khong phai 1
    duong thang chac chan)."""
    k = min(k, len(pool_feat))
    if k <= 0:
        z = np.zeros(pool_fwd_paths.shape[1] if pool_fwd_paths.ndim == 2 and pool_fwd_paths.shape[0] else 1)
        return z, z, z
    dist = np.linalg.norm(pool_feat - query_feat, axis=1)
    nn_idx = np.argpartition(dist, k - 1)[:k]
    neighbor_paths = pool_fwd_paths[nn_idx]  # (k, max_horizon)
    expected = neighbor_paths.mean(axis=0)
    lower = np.percentile(neighbor_paths, 25, axis=0)
    upper = np.percentile(neighbor_paths, 75, axis=0)
    return expected, lower, upper


# ---------------------------------------------------------------------------
# 4) Kiem dinh thong ke: hit-rate co THUC SU khac 50% hay chi la nhieu?
# ---------------------------------------------------------------------------

def _binomial_pvalue(k: int, n: int, p0: float = 0.5) -> float:
    """Kiem dinh nhi thuc 2 phia (xap xi chuan + hieu chinh lien tuc, KHONG
    can scipy - moi truong build sandbox khong co scipy) cho gia thuyet H0:
    ty le thanh cong THAT SU = p0 (mac dinh 50% = "khong hon gi ngau nhien").

    Tra ve p-value: xac suat quan sat duoc ket qua LECH KHOI ky vong bang
    hoac hon so voi thuc te, NEU H0 dung. p-value NHO (vd < 0.10) nghia la
    it kha nang chi la ngau nhien - co bang chung thong ke la ty le thanh
    cong that su KHAC 50%. Day la thay the cho nguong tuy y "+3 diem %" o
    ban thiet ke dau tien - co co so thong ke ro rang hon."""
    if n <= 0:
        return 1.0
    mean = n * p0
    std = math.sqrt(n * p0 * (1 - p0))
    if std == 0:
        return 1.0
    diff = max(abs(k - mean) - 0.5, 0.0)  # hieu chinh lien tuc (continuity correction)
    z = diff / std
    p_value = math.erfc(z / math.sqrt(2))
    return min(1.0, p_value)


# ---------------------------------------------------------------------------
# 5) Danh gia: 1 bo query/pool cho truoc (dung chung cho walk-forward)
# ---------------------------------------------------------------------------

@dataclass
class HorizonMetrics:
    horizon: int
    hit_rate: float          # % du doan dung HUONG (tang/giam) so voi thuc te
    mae: float                # sai so tuyet doi trung binh cua % return du bao
    n_samples: int


def _evaluate_queries(
    query_feat_std: np.ndarray,
    query_actual_paths: np.ndarray,
    pool_feat_std: np.ndarray,
    pool_paths: np.ndarray,
    k: int,
    horizons=DEFAULT_HORIZONS,
) -> dict[int, HorizonMetrics]:
    """Danh gia k-NN forecast cho 1 tap query (co san feature chuan hoa +
    outcome thuc te) so voi 1 pool hang xom cho truoc (co the la nhieu ma)."""
    hits = {h: [] for h in horizons}
    errors = {h: [] for h in horizons}
    if len(pool_feat_std) == 0 or len(query_feat_std) == 0:
        return {h: HorizonMetrics(h, float("nan"), float("nan"), 0) for h in horizons}

    for qi in range(len(query_feat_std)):
        expected, _, _ = _knn_forecast(query_feat_std[qi], pool_feat_std, pool_paths, k)
        actual_path = query_actual_paths[qi]
        for h in horizons:
            pred_h = expected[h - 1]
            actual_h = actual_path[h - 1]
            errors[h].append(abs(pred_h - actual_h))
            hits[h].append(1 if np.sign(pred_h) == np.sign(actual_h) else 0)

    metrics: dict[int, HorizonMetrics] = {}
    for h in horizons:
        n = len(errors[h])
        if n == 0:
            metrics[h] = HorizonMetrics(h, float("nan"), float("nan"), 0)
        else:
            metrics[h] = HorizonMetrics(h, float(np.mean(hits[h])), float(np.mean(errors[h])), n)
    return metrics


def _evaluate_baseline_queries(
    query_actual_paths: np.ndarray,
    pool_paths: np.ndarray,
    horizons=DEFAULT_HORIZONS,
) -> dict[int, HorizonMetrics]:
    """Du bao NGAY THO (ngay ngay tho = khong xet trang thai) - luon du bao
    CUNG 1 duong (trung binh duong di cua toan bo pool), bat ke trang thai
    hien tai. Dung de kiem tra k-NN co thuc su tot hon "trung binh chung"
    cua nhom/ma khong."""
    if len(pool_paths) == 0 or len(query_actual_paths) == 0:
        return {h: HorizonMetrics(h, float("nan"), float("nan"), 0) for h in horizons}

    baseline_path = pool_paths.mean(axis=0)
    hits = {h: [] for h in horizons}
    errors = {h: [] for h in horizons}
    for qi in range(len(query_actual_paths)):
        actual_path = query_actual_paths[qi]
        for h in horizons:
            pred_h = baseline_path[h - 1]
            actual_h = actual_path[h - 1]
            errors[h].append(abs(pred_h - actual_h))
            hits[h].append(1 if np.sign(pred_h) == np.sign(actual_h) else 0)

    metrics: dict[int, HorizonMetrics] = {}
    for h in horizons:
        n = len(errors[h])
        metrics[h] = HorizonMetrics(
            h,
            float(np.mean(hits[h])) if n else float("nan"),
            float(np.mean(errors[h])) if n else float("nan"),
            n,
        )
    return metrics


# ---------------------------------------------------------------------------
# 6) Walk-forward: nhieu fold lien tiep, gop ket qua + kiem dinh thong ke
# ---------------------------------------------------------------------------

def _make_walk_forward_folds(n_usable: int) -> list[tuple[int, int, int]]:
    """Tra ve list (val_start, oot_start, oot_end) - VI TRI (positional index)
    trong target_usable. Dung 'expanding window': train luon bat dau tu 0,
    cang ve fold sau train cang dai (thuc te hon 1 lan chia co dinh) - moi
    fold co 1 cua so OOT lien tiep, khong chong lap giua cac fold voi nhau."""
    init_end = int(n_usable * INIT_TRAIN_FRAC)
    remaining = n_usable - init_end

    if remaining < N_FOLDS * MIN_FOLD_VAL_LEN * 2:
        # qua it du lieu de chia nhieu fold - dung 1 fold duy nhat (fallback
        # an toan, tuong duong phuong phap 1-lan-chia truoc day).
        if remaining < MIN_FOLD_VAL_LEN * 2:
            return []
        val_len = max(int(n_usable * FOLD_VAL_FRAC), MIN_FOLD_VAL_LEN)
        val_len = min(val_len, remaining // 2)
        val_start = n_usable - remaining + (remaining - val_len) // 2
        oot_start = val_start + val_len
        if oot_start >= n_usable or val_start <= 0:
            return []
        return [(val_start, oot_start, n_usable)]

    fold_size = remaining // N_FOLDS
    val_len = max(int(n_usable * FOLD_VAL_FRAC), MIN_FOLD_VAL_LEN)
    folds = []
    for i in range(N_FOLDS):
        oot_start = init_end + i * fold_size
        oot_end = init_end + (i + 1) * fold_size if i < N_FOLDS - 1 else n_usable
        val_start = max(0, oot_start - val_len)
        if val_start <= 0 or oot_start <= val_start or oot_end <= oot_start:
            continue
        folds.append((val_start, oot_start, oot_end))
    return folds


def walk_forward_evaluate(
    target_usable: pd.DataFrame,
    pool_source: pd.DataFrame,
    k_grid=DEFAULT_K_GRID,
    horizons=DEFAULT_HORIZONS,
) -> dict:
    """Chay walk-forward qua nhieu fold (thay vi 1 lan chia Train/Val/OOT co
    dinh nhu ban dau) - moi fold: chon k tren VAL (hang xom tu pool, ngay <
    ngay bat dau VAL cua fold), roi danh gia OOT (hang xom tu pool, ngay <
    ngay bat dau OOT cua fold). Hang xom luon lay tu `pool_source` (co the
    la nhieu ma neu dung peer pooling) - LOC THEO NGAY (khong phai vi tri)
    de dam bao dong bo thoi gian giua cac ma khac nhau, tranh leakage cheo ma
    (vi du: khong bao gio dung du lieu 1 ma khac tai 1 ngay nam TRONG cua so
    OOT dang danh gia lam hang xom cho query cua ngay do).

    Tra ve dict:
      - oot_metrics / baseline_oot_metrics: GOP tat ca fold lai (tang co mau
        cho kiem dinh thong ke, dang tin cay hon 1 fold don le).
      - p_values: ket qua kiem dinh nhi thuc tren hit-rate DA GOP.
      - fold_hit_rates: hit-rate TUNG fold rieng (list) - de xem do ON DINH
        qua cac giai doan khac nhau (neu do lech giua cac fold lon, ket qua
        gop co the khong dang tin cay bang ve be ngoai).
      - k_live: k cua fold GAN NHAT (dung de du bao cho "hom nay").
      - n_folds_used: so fold thuc su chay duoc (co the < N_FOLDS neu thieu
        du lieu).
    """
    n_usable = len(target_usable)
    folds = _make_walk_forward_folds(n_usable)
    max_horizon = max(horizons)

    if not folds:
        nan_metrics = {h: HorizonMetrics(h, float("nan"), float("nan"), 0) for h in horizons}
        return {
            "oot_metrics": nan_metrics,
            "baseline_oot_metrics": nan_metrics,
            "p_values": {h: 1.0 for h in horizons},
            "fold_hit_rates": {h: [] for h in horizons},
            "k_live": k_grid[0],
            "n_folds_used": 0,
        }

    agg_hits = {h: 0 for h in horizons}
    agg_n = {h: 0 for h in horizons}
    agg_errsum = {h: 0.0 for h in horizons}
    base_hits = {h: 0 for h in horizons}
    base_errsum = {h: 0.0 for h in horizons}
    fold_hit_rates: dict[int, list[float]] = {h: [] for h in horizons}
    last_k = k_grid[0]

    dates = target_usable["date"].to_numpy()
    query_paths_all = np.stack(target_usable["fwd_path"].to_numpy())
    query_feat_all = target_usable[FEATURE_COLS].to_numpy()

    for (val_start, oot_start, oot_end) in folds:
        val_start_date = dates[val_start]
        oot_start_date = dates[oot_start]

        train_pool_raw = pool_source[pool_source["date"] < val_start_date]
        if len(train_pool_raw) < max(k_grid):
            continue
        train_feat_raw = train_pool_raw[FEATURE_COLS].to_numpy()
        feat_mean = train_feat_raw.mean(axis=0)
        feat_std = train_feat_raw.std(axis=0)

        train_feat_std = _standardize(train_feat_raw, feat_mean, feat_std)
        train_paths = np.stack(train_pool_raw["fwd_path"].to_numpy())

        val_query_feat_std = _standardize(query_feat_all[val_start:oot_start], feat_mean, feat_std)
        val_query_paths = query_paths_all[val_start:oot_start]

        best_k, best_score = k_grid[0], -np.inf
        for k in k_grid:
            m = _evaluate_queries(val_query_feat_std, val_query_paths, train_feat_std, train_paths, k, horizons)
            rates = [m[h].hit_rate for h in horizons if m[h].n_samples > 0 and not np.isnan(m[h].hit_rate)]
            score = float(np.mean(rates)) if rates else -np.inf
            if score > best_score:
                best_score, best_k = score, k

        trainval_pool_raw = pool_source[pool_source["date"] < oot_start_date]
        trainval_feat_std = _standardize(trainval_pool_raw[FEATURE_COLS].to_numpy(), feat_mean, feat_std)
        trainval_paths = np.stack(trainval_pool_raw["fwd_path"].to_numpy())

        oot_query_feat_std = _standardize(query_feat_all[oot_start:oot_end], feat_mean, feat_std)
        oot_query_paths = query_paths_all[oot_start:oot_end]

        fold_metrics = _evaluate_queries(oot_query_feat_std, oot_query_paths, trainval_feat_std, trainval_paths, best_k, horizons)
        fold_baseline = _evaluate_baseline_queries(oot_query_paths, trainval_paths, horizons)

        for h in horizons:
            m = fold_metrics[h]
            if m.n_samples > 0 and not np.isnan(m.hit_rate):
                agg_hits[h] += int(round(m.hit_rate * m.n_samples))
                agg_n[h] += m.n_samples
                agg_errsum[h] += m.mae * m.n_samples
                fold_hit_rates[h].append(m.hit_rate)
            b = fold_baseline[h]
            if b.n_samples > 0 and not np.isnan(b.hit_rate):
                base_hits[h] += int(round(b.hit_rate * b.n_samples))
                base_errsum[h] += b.mae * b.n_samples
        last_k = best_k

    oot_metrics: dict[int, HorizonMetrics] = {}
    baseline_oot_metrics: dict[int, HorizonMetrics] = {}
    p_values: dict[int, float] = {}
    for h in horizons:
        n = agg_n[h]
        if n > 0:
            hit_rate = agg_hits[h] / n
            mae = agg_errsum[h] / n
            oot_metrics[h] = HorizonMetrics(h, hit_rate, mae, n)
            p_values[h] = _binomial_pvalue(agg_hits[h], n, 0.5)
            base_rate = base_hits[h] / n
            base_mae = base_errsum[h] / n
            baseline_oot_metrics[h] = HorizonMetrics(h, base_rate, base_mae, n)
        else:
            oot_metrics[h] = HorizonMetrics(h, float("nan"), float("nan"), 0)
            baseline_oot_metrics[h] = HorizonMetrics(h, float("nan"), float("nan"), 0)
            p_values[h] = 1.0

    return {
        "oot_metrics": oot_metrics,
        "baseline_oot_metrics": baseline_oot_metrics,
        "p_values": p_values,
        "fold_hit_rates": fold_hit_rates,
        "k_live": last_k,
        "n_folds_used": len(folds),
    }


# ---------------------------------------------------------------------------
# Rationale (Buoc 5): mo ta trang thai bang tieng Viet
# ---------------------------------------------------------------------------

def _describe_state(row: pd.Series) -> str:
    parts = []

    if row["trend_score"] > 0.5:
        parts.append("giá đang ở trên EMA50 khá xa (xu hướng tăng)")
    elif row["trend_score"] < -0.5:
        parts.append("giá đang ở dưới EMA50 khá xa (xu hướng giảm)")
    else:
        parts.append("giá đang dao động quanh EMA50 (chưa rõ xu hướng)")

    if row["rsi14"] >= 70:
        parts.append(f"RSI({row['rsi14']:.0f}) ở vùng quá mua")
    elif row["rsi14"] <= 30:
        parts.append(f"RSI({row['rsi14']:.0f}) ở vùng quá bán")
    else:
        parts.append(f"RSI({row['rsi14']:.0f}) trung tính")

    if row["vol_ratio20"] >= 1.3:
        parts.append("khối lượng cao hơn trung bình 20 phiên")
    elif row["vol_ratio20"] <= 0.7:
        parts.append("khối lượng thấp hơn trung bình 20 phiên")
    else:
        parts.append("khối lượng ở mức bình thường")

    if row["beta_resid_20"] > 0.02:
        parts.append(f"có phần lệch riêng (đã trừ ảnh hưởng beta≈{row['beta60']:.1f} của VNINDEX) tích cực hơn thị trường trong 20 phiên qua")
    elif row["beta_resid_20"] < -0.02:
        parts.append(f"có phần lệch riêng (đã trừ ảnh hưởng beta≈{row['beta60']:.1f} của VNINDEX) tiêu cực hơn thị trường trong 20 phiên qua")
    else:
        parts.append(f"biến động gần như chỉ theo VNINDEX (beta≈{row['beta60']:.1f}), ít yếu tố riêng")

    if row["range_pct_120"] >= 0.85:
        parts.append("giá đang gần vùng đỉnh 120 phiên")
    elif row["range_pct_120"] <= 0.15:
        parts.append("giá đang gần vùng đáy 120 phiên")

    regime_vn = {
        "uptrend": "VNINDEX đang trong xu hướng tăng",
        "downtrend": "VNINDEX đang trong xu hướng giảm",
        "sideways": "VNINDEX đang đi ngang",
    }
    parts.append(regime_vn.get(row["market_regime"], ""))

    if row["adx14"] >= 25:
        parts.append(f"ADX({row['adx14']:.0f}) cho thấy xu hướng hiện tại khá mạnh")
    else:
        parts.append(f"ADX({row['adx14']:.0f}) cho thấy xu hướng chưa rõ ràng/yếu")

    return "; ".join(p for p in parts if p)


def _build_rationale(
    symbol: str,
    state_desc: str,
    oot_metrics: dict[int, HorizonMetrics],
    baseline_oot_metrics: dict[int, HorizonMetrics],
    p_values: dict[int, float],
    fold_hit_rates: dict[int, list[float]],
    peers_used: list[str],
    forecasts: dict[int, dict],
) -> str:
    lines = [f"Trạng thái hiện tại của {symbol}: {state_desc}."]
    if peers_used:
        lines.append(
            f"(Đã gộp lịch sử của {len(peers_used)} mã cùng nhóm để tăng cỡ mẫu/độ tin cậy "
            f"thống kê: {', '.join(peers_used)}.)"
        )
    else:
        lines.append(
            "(Chưa gộp được mã cùng nhóm ngành - kết quả dựa hoàn toàn vào lịch sử của riêng "
            "mã này, độ tin cậy thống kê có thể thấp hơn.)"
        )

    for h in DEFAULT_HORIZONS:
        f = forecasts[h]
        m = oot_metrics.get(h)
        p = p_values.get(h, 1.0)
        exp_ret = f["expected_return_pct"]
        direction = "tăng" if exp_ret > 0.3 else ("giảm" if exp_ret < -0.3 else "đi ngang")

        conf_txt = ""
        sig_txt = ""
        if m and m.n_samples > 0 and not np.isnan(m.hit_rate):
            fold_rates = fold_hit_rates.get(h, [])
            stability_txt = ""
            if len(fold_rates) > 1:
                stability_txt = f", ổn định qua {len(fold_rates)} giai đoạn (độ lệch {np.std(fold_rates)*100:.0f} điểm %)"
            conf_txt = f" (hit-rate {m.hit_rate*100:.0f}% trên {m.n_samples} lần kiểm tra ngoài mẫu/OOT{stability_txt})"

            if p < SIGNIFICANCE_ALPHA and m.hit_rate > 0.5:
                sig_txt = f" → CÓ ý nghĩa thống kê (p={p:.2f} < {SIGNIFICANCE_ALPHA}): khác 50% một cách đáng tin, không chỉ là ngẫu nhiên."
            elif p < SIGNIFICANCE_ALPHA and m.hit_rate < 0.5:
                sig_txt = f" → ⚠ CẢNH BÁO: hit-rate THẤP HƠN 50% có ý nghĩa thống kê (p={p:.2f}) - không nên tin theo chiều {direction} nêu trên."
            else:
                sig_txt = f" → CHƯA đủ bằng chứng thống kê (p={p:.2f} ≥ {SIGNIFICANCE_ALPHA}) để khẳng định khác 50%, có thể chỉ là nhiễu."

        lines.append(
            f"- {h} phiên tới: kỳ vọng {direction} khoảng {exp_ret:+.1f}% "
            f"(dải dự kiến {f['lower_pct']:+.1f}% đến {f['upper_pct']:+.1f}%){conf_txt}{sig_txt}"
        )

    lines.append(
        f"Lưu ý: kiểm định thống kê dùng nhị thức (binomial test) so với giả thuyết 'không có "
        f"tín hiệu riêng' (hit-rate=50%), ngưỡng p<{SIGNIFICANCE_ALPHA}, gộp kết quả qua nhiều "
        "giai đoạn walk-forward để tăng độ tin cậy - đây vẫn là dự báo thống kê dựa trên các "
        "giai đoạn lịch sử tương tự, không phải cam kết về giá, và vẫn có thể sai ngay cả khi "
        "có ý nghĩa thống kê."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator chinh (day la ham "phan tich" duoc goi tu UI)
# ---------------------------------------------------------------------------

@dataclass
class AssessmentResult:
    symbol: str
    as_of_date: str
    current_price: float
    calibrated_k: int
    oot_metrics: dict[int, HorizonMetrics]
    baseline_oot_metrics: dict[int, HorizonMetrics]      # du bao ngay tho (khong dieu kien) tren cung pool/OOT
    p_values: dict[int, float]                            # kiem dinh nhi thuc: hit-rate co khac 50% co y nghia khong
    fold_hit_rates: dict[int, list[float]]                 # hit-rate TUNG fold walk-forward (do on dinh)
    n_folds_used: int
    peers_used: list[str]                                  # ma cung nhom da duoc gop lam hang xom
    forecasts: dict[int, dict]     # horizon -> {expected_path, lower_path, upper_path, price_path, expected_return_pct, lower_pct, upper_pct}
    state_description: str
    rationale_text: str
    warnings: list[str] = field(default_factory=list)


def generate_assessment(
    symbol: str,
    stock_df: pd.DataFrame,
    vnindex_df: pd.DataFrame,
    peer_price_dfs: Optional[dict[str, pd.DataFrame]] = None,
    horizons=DEFAULT_HORIZONS,
    k_grid=DEFAULT_K_GRID,
) -> AssessmentResult:
    """Ham phan tich chinh (Buoc 3-5). Nhan vao gia lich su cua 1 ma (tu
    data_manager.get_price_df), cua VNINDEX, va (tuy chon) gia lich su cua
    cac ma CUNG NHOM NGANH (peer_price_dfs, xem sector_map.py) de gop lam
    hang xom k-NN - tang co mau/do tin cay thong ke. Tra ve AssessmentResult
    day du: trang thai hien tai, du bao 3/10/20/50 phien (nhieu doan, co dai
    tin cay), do tin cay OOT (walk-forward + kiem dinh thong ke), va nhan
    dinh bang van ban."""
    warnings: list[str] = []
    max_horizon = max(horizons)

    built = build_pool_source(symbol, stock_df, vnindex_df, peer_price_dfs, max_horizon)
    target_full = built["target_full"]
    target_usable = built["target_usable"]
    pool_source = built["pool_source"]
    peers_used = built["peers_used"]

    if len(target_usable) < MIN_USABLE_ROWS:
        warnings.append(
            f"Chỉ có {len(target_usable)} dòng dữ liệu hợp lệ sau khi bỏ warm-up - ít hơn mức "
            f"khuyến nghị ({MIN_USABLE_ROWS}). Kết quả hiệu chỉnh/OOT có thể không ổn định."
        )
    if not peers_used:
        warnings.append(
            "Chưa gộp được mã cùng nhóm ngành (không có trong danh sách tham khảo hoặc thiếu "
            "dữ liệu) - kết quả dựa hoàn toàn vào lịch sử của riêng mã này, độ tin cậy thống kê "
            "có thể thấp hơn."
        )

    wf = walk_forward_evaluate(target_usable, pool_source, k_grid, horizons)
    if wf["n_folds_used"] == 0:
        warnings.append("Không đủ dữ liệu để chạy walk-forward nhiều giai đoạn - kết quả có thể không ổn định.")
    k_live = wf["k_live"]

    # --- du bao cho "hien tai" (dong cuoi cung cua target_full) ---
    today_date = target_full["date"].iloc[-1]
    live_pool_raw = pool_source[pool_source["date"] < today_date]
    if len(live_pool_raw) < k_live:
        warnings.append("Không đủ dữ liệu lịch sử để làm 'hàng xóm' dự báo - kết quả dự báo có thể không đáng tin.")

    if len(live_pool_raw) > 0:
        live_feat_raw = live_pool_raw[FEATURE_COLS].to_numpy()
        feat_mean = live_feat_raw.mean(axis=0)
        feat_std = live_feat_raw.std(axis=0)
        pool_feat_std = _standardize(live_feat_raw, feat_mean, feat_std)
        pool_paths = np.stack(live_pool_raw["fwd_path"].to_numpy())
    else:
        feat_mean = np.zeros(len(FEATURE_COLS))
        feat_std = np.ones(len(FEATURE_COLS))
        pool_feat_std = np.empty((0, len(FEATURE_COLS)))
        pool_paths = np.empty((0, max_horizon))

    q_feat_raw = target_full[FEATURE_COLS].to_numpy()[-1]
    q_feat_std = _standardize(q_feat_raw, feat_mean, feat_std)
    current_price = float(target_full["close"].to_numpy()[-1])

    if len(pool_feat_std) > 0:
        expected, lower, upper = _knn_forecast(q_feat_std, pool_feat_std, pool_paths, k_live)
    else:
        expected = lower = upper = np.zeros(max_horizon)

    forecasts = {}
    for h in horizons:
        exp_path_pct = (expected[:h] * 100).tolist()
        low_path_pct = (lower[:h] * 100).tolist()
        up_path_pct = (upper[:h] * 100).tolist()
        price_path = [current_price * (1 + r) for r in expected[:h]]
        lower_price_path = [current_price * (1 + r) for r in lower[:h]]
        upper_price_path = [current_price * (1 + r) for r in upper[:h]]
        forecasts[h] = {
            "expected_return_path_pct": exp_path_pct,
            "lower_return_path_pct": low_path_pct,
            "upper_return_path_pct": up_path_pct,
            "price_path": price_path,
            "lower_price_path": lower_price_path,
            "upper_price_path": upper_price_path,
            "expected_return_pct": exp_path_pct[-1],
            "lower_pct": low_path_pct[-1],
            "upper_pct": up_path_pct[-1],
        }

    state_desc = _describe_state(target_full.iloc[-1])
    rationale = _build_rationale(
        symbol, state_desc, wf["oot_metrics"], wf["baseline_oot_metrics"], wf["p_values"],
        wf["fold_hit_rates"], peers_used, forecasts,
    )

    as_of_date = today_date
    as_of_date_str = as_of_date.strftime("%Y-%m-%d") if hasattr(as_of_date, "strftime") else str(as_of_date)

    return AssessmentResult(
        symbol=symbol,
        as_of_date=as_of_date_str,
        current_price=current_price,
        calibrated_k=k_live,
        oot_metrics=wf["oot_metrics"],
        baseline_oot_metrics=wf["baseline_oot_metrics"],
        p_values=wf["p_values"],
        fold_hit_rates=wf["fold_hit_rates"],
        n_folds_used=wf["n_folds_used"],
        peers_used=peers_used,
        forecasts=forecasts,
        state_description=state_desc,
        rationale_text=rationale,
        warnings=warnings,
    )
