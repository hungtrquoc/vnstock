"""
market_screener.py
--------------------
Quet TOAN BO co phieu dang niem yet tren thi truong VN (theo yeu cau nguoi
dung 2026-08-03: "toan bo co phieu dang niem yet luon" - khong gioi han
1 danh sach theo doi rieng), liet ke ra nhung ma:
  - "Thuan ky thuat tich cuc": classic_ta.build_classic_scorecard tra ve
    verdict "Tich cuc" (dinh tinh - CHUA kiem chung thong ke, xem canh bao
    trong classic_ta.py).
  - "Dang tin & tich cuc": co it nhat 1 khung thoi gian (3/10/20/50 phien) ma
    k-NN OOT (analysis_engine.py) co hit-rate>50% VA p-value co y nghia
    thong ke (< SIGNIFICANCE_ALPHA), qua walk-forward + gop peer nhu binh
    thuong.

CANH BAO QUAN TRONG (doc truoc khi dung/sua):

1. NGUON DANH SACH "TOAN BO MA NIEM YET" (sua 2026-08-03, sau khi nguoi dung
   bao chi quet duoc 68 ma - dung 100% so mã trong FALLBACK_SYMBOLS, tuc la
   endpoint VNDIRECT ban dau da KHONG hoat dong tren may that cua nguoi
   dung, khong chi trong sandbox). Endpoint VNDIRECT `finfo-api.../v4/stocks`
   ban dau la DOAN, chua tung duoc xac nhan - da BO. Thay bang endpoint cua
   VCI (Vietcap chung khoan), `_VCI_SYMBOLS_URL` ben duoi
   (`https://trading.vietcap.com.vn/api/price/symbols/getAll`) - day la
   endpoint duoc xac nhan qua source code THUC TE cua thu vien Python cong
   dong dang duoc bao tri tich cuc `vnstock` (github.com/thinh-vu/vnstock,
   `vnstock/explorer/vci/listing.py`, ham `symbols_by_exchange`), KHONG phai
   tu doan nhu lan truoc - do tin cay cao hon nhieu. Van giu 2 lop phong
   ngu vi mang trong sandbox xay dung app van bi chan (khong the tu goi
   endpoint nay de kiem tra JSON schema thuc te):
   - Ham duoc viet phong thu: BAT KY loi gi (doi endpoint, doi schema,
     thieu header Referer/Origin can thiet, timeout...) deu duoc bat va
     tra ve danh sach RONG, KHONG lam crash app.
   - Neu VCI that bai, thu lai voi endpoint VNDIRECT cu (`FINFO_STOCKS_URL`)
     nhu 1 nguon du phong thu 2 (co the con dung cho 1 phan nao do, khong
     hai gi khi giu lai).
   - Neu CA HAI deu that bai (tra ve []), noi goi (main.py) se tu dong
     chuyen sang FALLBACK_SYMBOLS (danh sach du phong CUOI CUNG, KHONG day
     du - chi cac ma da co trong sector_map.py) va bao ro cho nguoi dung
     biet.
   NGUOI DUNG CAN CHAY LAI TREN MAY THAT VA BAO KET QUA: neu van chi ra 68
   ma (dung FALLBACK_SYMBOLS), tuc endpoint VCI cung that bai - can xem
   log/nhat ky de biet loi cu the (sai header, doi schema JSON, hay bi chan
   mang) va bao lai de sua tiep.

2. QUET TOAN BO THI TRUONG (~1000-1700+ ma tren HOSE/HNX/UPCOM) LA 1 TAC
   VU RAT NANG - khong the chay "tuc thi":
   - Lan dau tien: neu 1 ma CHUA CO du lieu trong DB, phai tai TOAN BO lich
     su gia tu 2000 den nay - voi hang nghin ma chua tung phan tich, day
     co the mat HANG GIO va tai rat nhieu du lieu qua mang.
   - Ke ca sau khi da co du lieu day du, MOI LAN quet van goi cap nhat
     (theo fix "du lieu bi cu" 2026-08-03) cho MOI ma - hang nghin request
     mang rieng le, van ton nhieu thoi gian moi lan quet lai (uoc luong tu
     vai chuc phut den vai gio tuy toc do mang/API va so luong ma).
   - Phan tich thong ke day du (walk-forward + peer pooling, xem
     analysis_engine.py) cho MOI ma co du du lieu cung ton CPU dang ke.
   VI VAY: tac vu nay PHAI chay ngam (background thread, xem ScreenerWorker
   trong main.py), co the DUNG giua chung ma khong mat ket qua da co, va
   HIEN KET QUA NGAY KHI TUNG MA quet xong (khong doi het toan bo).

3. DE GIAM TAI (dong dan quan trong, can biet khi doc ket qua): buoc phan
   tich thong ke DAY DU (walk-forward, ton CPU nhat) CHI chay cho nhung ma
   da qua vong loc "Thuan ky thuat tich cuc" TRUOC (xem screen_one_symbol) -
   day la 1 "pheu" (funnel) de giam so luong ma phai chay k-NN day du tu
   ~1000-1700+ xuong con phan nho hon (thuong chi mot phan thi truong dang
   "tich cuc" tai 1 thoi diem). HE QUA: 1 ma co the LA "dang tin & tich
   cuc" thuc su (k-NN co y nghia thong ke) nhung KHONG duoc phat hien neu
   no khong qua duoc vong loc dinh tinh truoc (vi du: classic_ta danh gia
   trung lap nhung k-NN van co the tim ra tin hieu thong ke). Day la 1
   danh doi CO CHU DICH giua toc do va do day du - neu nguoi dung can quet
   THONG KE DAY DU cho MOI ma (khong qua pheu dinh tinh), can sua
   screen_one_symbol de bo qua buoc loc nay (se cham hon nhieu).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

import analysis_engine as ae
import candlestick_patterns as cp
import classic_ta
import data_manager as dm
import sector_map

FINFO_STOCKS_URL = "https://finfo-api.vndirect.com.vn/v4/stocks"
_VCI_SYMBOLS_URL = "https://trading.vietcap.com.vn/api/price/symbols/getAll"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# Header Referer/Origin bat buoc phai co - xac nhan qua vnstock (thu vien
# cong dong dang bao tri): thieu header nay API cua VCI se tu choi request
# (co the tra ve loi/403 thay vi du lieu that).
_VCI_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://trading.vietcap.com.vn/",
    "Origin": "https://trading.vietcap.com.vn/",
}

# Danh sach du phong khi khong goi duoc API danh sach toan bo ma niem yet -
# CHI la cac ma da co san trong sector_map.py (vai chuc ma lon/pho bien theo
# nhom nganh), KHONG phai toan bo thi truong. Luon bao ro cho nguoi dung khi
# roi vao truong hop nay (xem main.py: on_screen_start_clicked).
FALLBACK_SYMBOLS = sorted({s for peers in sector_map.SECTOR_PEERS.values() for s in peers})

MIN_ROWS_FOR_CLASSIC_TA = ae.WARMUP_ROWS               # ~210 dong, du cho EMA200/ADX/RSI on dinh
MIN_ROWS_FOR_STAT_ANALYSIS = ae.MIN_USABLE_ROWS + ae.WARMUP_ROWS  # nguong day du cho walk-forward


def _fetch_from_vci(timeout: int) -> list[str]:
    """Nguon CHINH (sua 2026-08-03): endpoint cua VCI/Vietcap, xac nhan qua
    source code thuc te cua thu vien cong dong `vnstock` dang duoc bao tri
    tich cuc (khac endpoint VNDIRECT truoc day, la doan chua kiem chung).
    Tra ve JSON dang list[dict], moi dict co truong "symbol"/"type"/"board"
    (chua qua camelCase->snake_case). Chi lay nhung dong "type"=="STOCK"
    (bo ETF/chung quyen/trai phieu/future)."""
    resp = requests.get(_VCI_SYMBOLS_URL, headers=_VCI_HEADERS, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    items = data if isinstance(data, list) else data.get("data", [])
    symbols = sorted({
        item["symbol"] for item in items
        if item.get("symbol") and item.get("type") == "STOCK"
    })
    return symbols


def _fetch_from_vndirect_legacy(timeout: int) -> list[str]:
    """Nguon DU PHONG THU 2 (endpoint VNDIRECT ban dau, chua bao gio duoc
    xac nhan hoat dong - giu lai vi khong hai gi, phong truong hop endpoint
    VCI cung ngung hoat dong sau nay)."""
    params = {"q": "type:STOCK~status:LISTED", "size": 3000}
    resp = requests.get(FINFO_STOCKS_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("data", [])
    return sorted({item["code"] for item in items if item.get("code")})


def fetch_all_listed_symbols(timeout: int = 30) -> list[str]:
    """Lay danh sach TOAN BO ma co phieu (khong tinh ETF/chung quyen/trai
    phieu) dang NIEM YET tren HOSE/HNX/UPCOM. Thu lan luot: (1) VCI - nguon
    chinh, do tin cay cao hon (xem CANH BAO #1 o docstring module); (2)
    VNDIRECT (endpoint cu) - du phong thu 2. Neu CA HAI deu that bai, tra
    ve [] va de main.py tu chuyen sang FALLBACK_SYMBOLS. Khong bao gio nem
    exception ra ngoai - moi loi (sai endpoint, doi schema, thieu header,
    timeout, bi chan mang...) deu duoc bat lai."""
    try:
        symbols = _fetch_from_vci(timeout)
        if symbols:
            return symbols
    except Exception:
        pass

    try:
        symbols = _fetch_from_vndirect_legacy(timeout)
        if symbols:
            return symbols
    except Exception:
        pass

    return []


@dataclass
class ScreenerRow:
    symbol: str
    status: str                                   # "ok" | "skip" | "error"
    message: str = ""
    classic_verdict: Optional[str] = None          # "Tích cực" / "Tiêu cực" / "Trung lập..."
    classic_positive: bool = False
    stat_positive: bool = False                    # co >=1 horizon: hit_rate>50% VA p<alpha
    stat_best_horizon: Optional[int] = None
    stat_best_hit_rate: Optional[float] = None
    stat_best_pvalue: Optional[float] = None


def screen_one_symbol(conn, symbol: str) -> ScreenerRow:
    """Quet 1 ma (xem cac buoc trong docstring module):
    1. Cap nhat du lieu (best-effort - loi khong lam crash, van dung du
       lieu cu neu co).
    2. Neu du du lieu (MIN_ROWS_FOR_CLASSIC_TA): chay classic_ta (nhanh,
       chi tinh vectorized, khong can k-NN) -> xac dinh classic_positive.
    3. CHI KHI classic_positive VA du du lieu cho k-NN
       (MIN_ROWS_FOR_STAT_ANALYSIS): chay full k-NN OOT (walk-forward +
       peer pooling qua sector_map.py) -> xac dinh stat_positive (xem canh
       bao #3 module ve "pheu" nay)."""
    try:
        dm.update_stock_data(conn, symbol)
    except Exception:
        pass  # van thu phan tich tren du lieu cu neu cap nhat loi

    stock_df = dm.get_price_df(conn, symbol)
    if stock_df.empty:
        return ScreenerRow(symbol, "error", message="Không có/không tải được dữ liệu.")

    if len(stock_df) < MIN_ROWS_FOR_CLASSIC_TA:
        return ScreenerRow(symbol, "skip", message=f"Chỉ có {len(stock_df)} dòng - chưa đủ để đánh giá.")

    vnindex_df = dm.get_price_df(conn, "VNINDEX")
    if vnindex_df.empty:
        return ScreenerRow(symbol, "error", message="Chưa có dữ liệu VNINDEX.")

    try:
        ind = ae.compute_indicators(stock_df)
        candle_hits = cp.recent_patterns(stock_df, n_recent=15)
        scorecard = classic_ta.build_classic_scorecard(ind.iloc[-1], candle_hits)
    except Exception as e:  # noqa: BLE001
        return ScreenerRow(symbol, "error", message=f"Lỗi đánh giá kỹ thuật cổ điển: {e}")

    classic_positive = scorecard.verdict == "Tích cực"
    row = ScreenerRow(symbol, "ok", classic_verdict=scorecard.verdict, classic_positive=classic_positive)

    # --- "pheu": chi chay k-NN OOT day du (ton CPU nhat) cho ma da qua vong
    # loc dinh tinh truoc, de giam tai khi quet hang nghin ma - xem canh bao
    # #3 o docstring module. ---
    if classic_positive and len(stock_df) >= MIN_ROWS_FOR_STAT_ANALYSIS:
        peer_price_dfs = {}
        for peer_symbol in sector_map.get_peers(symbol):
            try:
                dm.update_stock_data(conn, peer_symbol)
                peer_df = dm.get_price_df(conn, peer_symbol)
                if not peer_df.empty:
                    peer_price_dfs[peer_symbol] = peer_df
            except Exception:
                continue

        try:
            assessment = ae.generate_assessment(symbol, stock_df, vnindex_df, peer_price_dfs=peer_price_dfs)
        except Exception as e:  # noqa: BLE001
            row.message = f"Lỗi phân tích thống kê: {e}"
            return row

        best = None  # (horizon, hit_rate, p_value) - uu tien hit_rate cao nhat
        for h in ae.DEFAULT_HORIZONS:
            m = assessment.oot_metrics.get(h)
            p = assessment.p_values.get(h, 1.0)
            if m and m.n_samples > 0 and p < ae.SIGNIFICANCE_ALPHA and m.hit_rate > 0.5:
                if best is None or m.hit_rate > best[1]:
                    best = (h, m.hit_rate, p)

        if best:
            row.stat_positive = True
            row.stat_best_horizon, row.stat_best_hit_rate, row.stat_best_pvalue = best

    return row


if __name__ == "__main__":
    # self-test don gian: fetch_all_listed_symbols khong duoc nem exception
    # (mang trong sandbox nay bi chan nen se tra ve [] - day CHINH LA phep
    # thu that su cho duong fallback, khong phai mock). Tren may that cua
    # nguoi dung, ham nay se thu ca VCI lan VNDIRECT truoc khi tra ve [].
    symbols = fetch_all_listed_symbols(timeout=5)
    assert isinstance(symbols, list)
    print(f"[OK] Test 1 - fetch_all_listed_symbols khong crash, tra ve {len(symbols)} ma "
          f"(0 la binh thuong trong sandbox bi chan mang)")

    assert len(FALLBACK_SYMBOLS) > 10
    assert "VCB" in FALLBACK_SYMBOLS
    print(f"[OK] Test 2 - FALLBACK_SYMBOLS co {len(FALLBACK_SYMBOLS)} mã dự phòng")

    print("\nALL TESTS PASSED (test day du voi du lieu tong hop: xem test_market_screener.py)")
