"""
sector_map.py
-------------
Danh sach nhom nganh/beta tinh (static) cho co phieu VN - dung lam "peer
group" de gop du lieu lich su khi phan tich k-NN (xem analysis_engine.py:
build_pool_source). VNDIRECT price API khong tra ve nganh/ICB, nen day la
giai phap thuc te thay the: 1 bang tra cuu thu cong, du de tang co mau cho
kiem dinh thong ke ma khong can 1 nguon du lieu nganh rieng.

Neu 1 ma khong co trong danh sach nao (vd ma le, ma moi len san), get_peers
tra ve [] - analysis_engine se tu dong lui ve che do phan tich rieng 1 ma
(an toan, khong loi, chi la khong duoc loi ich tang co mau).

Danh sach co the khong day du/khong con cap nhat - chi mang tinh tham khao,
KHONG phai phan loai ICB chinh thuc.
"""

SECTOR_PEERS: dict[str, list[str]] = {
    "Ngan hang": [
        "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB",
        "HDB", "TPB", "SHB", "EIB", "LPB", "OCB", "VIB", "MSB",
    ],
    "Chung khoan": [
        "SSI", "VND", "HCM", "VCI", "MBS", "SHS", "VIX", "FTS", "BSI", "CTS",
    ],
    "Bat dong san": [
        "VHM", "VIC", "NVL", "PDR", "DXG", "KDH", "NLG", "DIG",
        "HDG", "CEO", "SCR", "IJC", "VRE",
    ],
    "Thep": [
        "HPG", "HSG", "NKG", "TVN", "SMC", "TLH", "VGS",
    ],
    "Ban le": [
        "MWG", "PNJ", "FRT", "DGW", "PET",
    ],
    "Dau khi": [
        "GAS", "PLX", "PVD", "PVS", "PVT", "BSR", "OIL", "PVC",
    ],
    "Xay dung": [
        "CTD", "HBC", "VCG", "FCN", "HHV", "LCG",
    ],
    "Hang khong": [
        "HVN", "VJC", "ACV",
    ],
}

MAX_PEERS = 6


def get_peers(symbol: str, max_peers: int = MAX_PEERS) -> list[str]:
    """Tra ve danh sach ma CUNG NHOM voi `symbol` (khong bao gom chinh no),
    toi da `max_peers` ma. Neu khong tim thay nhom nao chua `symbol`, tra ve
    [] (khong loi - noi goi phai tu xu ly truong hop khong co peer)."""
    symbol = symbol.upper().strip()
    for peers in SECTOR_PEERS.values():
        if symbol in peers:
            others = [p for p in peers if p != symbol]
            return others[:max_peers]
    return []


if __name__ == "__main__":
    # self-test don gian
    assert "BID" in get_peers("VCB")
    assert "VCB" not in get_peers("VCB")
    assert len(get_peers("VCB")) <= MAX_PEERS
    assert get_peers("ZZZ_NOT_A_REAL_TICKER") == []
    print("sector_map.py self-test: OK")
