"""
audit_hitrate.py
-----------------
Cong cu chan doan: quet TAT CA cac ma dang co du lieu trong DB, chay
generate_assessment cho tung ma (co gop peer neu co trong sector_map), va in
ra 1 bang so sanh hit-rate cua model (k-NN) so voi hit-rate baseline (ngay
tho) + p-value cho ca 3 khung thoi gian (3/10/20 phien).

LY DO CAN CONG CU NAY: nguoi dung bao hit-rate < 50% o HAU HET cac ma da
chay - neu chi la "khong co tin hieu" (nhieu ngau nhien), ve ly thuyet hit-
rate phai dao dong QUANH 50% (nua so ma tren 50%, nua duoi 50%), chu KHONG
phai luon luon duoi 50%. Luon duoi 50% o nhieu ma la dau hieu dang ngo:
  (a) co the la LOI ky thuat (vd sai lech huong/off-by-one o dau do trong
      pipeline lam du bao bi DAO NGUOC so voi thuc te), hoac
  (b) co the la mot HIEN TUONG THAT: dac diem "dao chieu ngan han" (mean-
      reversion) cua co phieu VN trong giai doan du lieu nay - neu vay, mo
      hinh dang "theo da" (momentum-following qua cac dac trung trend_score/
      RSI/MACD) se he thong sai huong, va NGHICH DAO tin hieu co the lai co
      hit-rate > 50%.

CACH DOC KET QUA (quan trong nhat khi chan doan):
  - Neu Baseline (du bao ngay tho, khong xet trang thai) CUNG duoi 50% o
    nhieu ma -> nghieng ve (a) loi ky thuat hoac dac diem du lieu chung, vi
    baseline khong dua vao feature/model gi ca, chi la trung binh duong di.
  - Neu Baseline binh thuong (~50% hoac cao hon) nhung k-NN LUON THAP HON
    Baseline -> nghieng ve (b): cac feature dang "chi sai huong" (feature
    conditioning dang phan tac dung) - dang chu y va co the can dieu chinh
    trong so/dac trung, hoac thu nghiem dao nguoc tin hieu de kiem tra.

Chay: python audit_hitrate.py [--csv out.csv] [--min-rows 610]
(mac dinh doc DB tai %USERPROFILE%\\VNStockApp\\vn_stock_data.db, giong main.py)
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_manager as dm
import analysis_engine as ae
import sector_map

DB_PATH = Path.home() / "VNStockApp" / "vn_stock_data.db"


def list_symbols(conn) -> list[str]:
    cur = conn.execute("SELECT DISTINCT symbol FROM daily_prices ORDER BY symbol")
    return [row[0] for row in cur.fetchall() if row[0] != "VNINDEX"]


def audit(conn, min_rows: int) -> list[dict]:
    symbols = list_symbols(conn)
    vnindex_df = dm.get_price_df(conn, "VNINDEX")
    if vnindex_df.empty:
        print("LỖI: chưa có dữ liệu VNINDEX trong DB - hãy chạy 'Cập nhật dữ liệu' cho VNINDEX trước.")
        return []

    rows = []
    for symbol in symbols:
        stock_df = dm.get_price_df(conn, symbol)
        if len(stock_df) < min_rows:
            print(f"[bỏ qua] {symbol}: chỉ có {len(stock_df)} dòng (< {min_rows})")
            continue

        peer_price_dfs = {}
        for peer_symbol in sector_map.get_peers(symbol):
            try:
                peer_df = dm.get_price_df(conn, peer_symbol)
                if not peer_df.empty:
                    peer_price_dfs[peer_symbol] = peer_df
            except Exception:
                continue

        try:
            result = ae.generate_assessment(symbol, stock_df, vnindex_df, peer_price_dfs=peer_price_dfs)
        except Exception as e:  # noqa: BLE001
            print(f"[lỗi] {symbol}: {e}")
            continue

        for h in ae.DEFAULT_HORIZONS:
            m = result.oot_metrics.get(h)
            b = result.baseline_oot_metrics.get(h)
            p = result.p_values.get(h)
            if not m or m.n_samples == 0:
                continue
            rows.append({
                "symbol": symbol,
                "horizon": h,
                "hit_rate_model": round(m.hit_rate, 3),
                "hit_rate_baseline": round(b.hit_rate, 3) if b and b.n_samples else None,
                "p_value": round(p, 3) if p is not None else None,
                "n_samples": m.n_samples,
                "n_folds": result.n_folds_used,
                "n_peers": len(result.peers_used),
            })
        print(f"[OK] {symbol}: k={result.calibrated_k} peers={result.peers_used} folds={result.n_folds_used}")

    return rows


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("Không có kết quả nào để tổng hợp.")
        return

    print("\n" + "=" * 100)
    print(f"{'Mã':<8}{'Phiên':>7}{'Hit-rate model':>16}{'Baseline':>12}{'p-value':>10}{'n mẫu':>8}{'folds':>7}{'peers':>7}")
    print("-" * 100)
    for r in rows:
        base_txt = f"{r['hit_rate_baseline']*100:.0f}%" if r["hit_rate_baseline"] is not None else "n/a"
        print(
            f"{r['symbol']:<8}{r['horizon']:>7}{r['hit_rate_model']*100:>15.0f}%{base_txt:>12}"
            f"{r['p_value']:>10}{r['n_samples']:>8}{r['n_folds']:>7}{r['n_peers']:>7}"
        )
    print("=" * 100)

    # --- tong hop de chan doan nhanh ---
    for h in ae.DEFAULT_HORIZONS:
        sub = [r for r in rows if r["horizon"] == h]
        if not sub:
            continue
        n = len(sub)
        n_model_below_50 = sum(1 for r in sub if r["hit_rate_model"] < 0.5)
        n_base_below_50 = sum(1 for r in sub if r["hit_rate_baseline"] is not None and r["hit_rate_baseline"] < 0.5)
        n_model_below_base = sum(
            1 for r in sub if r["hit_rate_baseline"] is not None and r["hit_rate_model"] < r["hit_rate_baseline"]
        )
        avg_model = sum(r["hit_rate_model"] for r in sub) / n
        print(
            f"\nHorizon={h}: {n} mã | model<50%: {n_model_below_50}/{n} | "
            f"baseline<50%: {n_base_below_50}/{n} | model<baseline: {n_model_below_base}/{n} | "
            f"trung bình hit-rate model: {avg_model*100:.1f}%"
        )

    print(
        "\nGợi ý đọc: nếu 'baseline<50%' cũng cao gần bằng 'model<50%' -> nghiêng về vấn đề dữ liệu/kỹ "
        "thuật chung (kiểm tra code), không phải riêng model. Nếu 'model<baseline' cao (model luôn tệ "
        "hơn baseline) trong khi baseline bình thường -> các đặc trưng đang chỉ sai hướng một cách có hệ "
        "thống - đáng để xem lại trọng số/đặc trưng đang dùng."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Lưu kết quả chi tiết ra file CSV")
    parser.add_argument("--min-rows", type=int, default=ae.MIN_USABLE_ROWS + ae.WARMUP_ROWS)
    args = parser.parse_args()

    conn = dm.get_connection(DB_PATH)
    rows = audit(conn, args.min_rows)
    print_summary(rows)

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nĐã lưu chi tiết ra {args.csv}")


if __name__ == "__main__":
    main()
