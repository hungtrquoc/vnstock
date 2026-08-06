"""
main.py - VN Stock App (Phase 1: Data pipeline UI)
----------------------------------------------------
Native desktop app (PyQt6). Phase 1 chi lam Buoc 1-2 cua flow:
  - Nhap ma co phieu
  - Cap nhat du lieu (thong minh: full pull neu chua co, incremental neu da co)
  - Keo lai toan bo du lieu (dung khi co phieu chia co tuc/tach)
  - Xem preview du lieu da luu trong DB

Phan tich ky thuat (Buoc 3-5) se duoc them o Phase 2/3, trong module rieng
(analysis_engine.py) va se duoc goi tu day nhu 1 tab/man hinh moi - khong
dung chung code voi phan lay du lieu nay.
"""
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import html as _html
import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QPlainTextEdit,
    QTextEdit,
    QMessageBox,
    QTabWidget,
    QScrollArea,
    QFrame,
    QHeaderView,
    QCheckBox,
    QFileDialog,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import data_manager as dm
import analysis_engine as ae
import chart_view as cv
import candlestick_patterns as cp
import classic_ta
import sector_map
import market_screener as screener

APP_DIR = Path(__file__).resolve().parent
# QUAN TRONG: KHONG dat DB trong folder Google Drive / OneDrive dong bo (nhu
# G:\My Drive\...). SQLite ghi lien tuc + file .db-wal/.db-shm rat de bi loi
# khoa file hoac xung dot dong bo tren cac folder cloud-sync. Luu o thu muc
# local rieng tren may (%USERPROFILE%\VNStockApp) thay vi canh file code.
DB_PATH = Path.home() / "VNStockApp" / "vn_stock_data.db"


def _summarize_oot_html(result: "ae.AssessmentResult") -> str:
    """Tom TAT 1 dong duy nhat (khong lap lai so lieu da co trong bang) cho
    phan du bao da kiem chung OOT - theo phan hoi nguoi dung "format van
    xau, nhieu chu, khong tom gon lai van de": chi dua ra 1 KET LUAN duy
    nhat, dua tren KIEM DINH THONG KE (p-value nhi thuc so voi 50%, xem
    analysis_engine._binomial_pvalue) thay vi nguong tuy y "+3 diem %" ban
    dau - de tranh goi y "co tin hieu" khi chenh lech nho co the chi la
    nhieu ngau nhien. Neu co horizon nao hit_rate<50% CO Y NGHIA thong ke
    thi uu tien canh bao do truoc tien (nguy hiem hon la bo lo co hoi)."""
    horizons = sorted(result.oot_metrics.keys())
    warn_h = None       # horizon co hit_rate<50% VA co y nghia thong ke - canh bao truoc tien
    sig_positive = None  # (horizon, hit_rate, p_value) - horizon co hit_rate>50% VA co y nghia, hit_rate cao nhat
    for h in horizons:
        m = result.oot_metrics.get(h)
        p = result.p_values.get(h, 1.0)
        if not m or m.n_samples == 0 or np.isnan(m.hit_rate):
            continue
        if p < ae.SIGNIFICANCE_ALPHA and m.hit_rate < 0.5 and warn_h is None:
            warn_h = h
        if p < ae.SIGNIFICANCE_ALPHA and m.hit_rate > 0.5:
            if sig_positive is None or m.hit_rate > sig_positive[1]:
                sig_positive = (h, m.hit_rate, p)

    if warn_h is not None:
        m = result.oot_metrics[warn_h]
        p = result.p_values[warn_h]
        color = "#c62828"
        headline = (
            f"KHÔNG ĐÁNG TIN ở {warn_h} phiên — hit-rate {m.hit_rate*100:.0f}% thấp hơn 50% "
            f"CÓ ý nghĩa thống kê (p={p:.2f})"
        )
    elif sig_positive is not None:
        h, hit, p = sig_positive
        exp_ret = result.forecasts[h]["expected_return_pct"]
        direction = "TĂNG" if exp_ret > 0.3 else ("GIẢM" if exp_ret < -0.3 else "ĐI NGANG")
        color = "#2e7d32"
        headline = f"CÓ TÍN HIỆU {direction} ở {h} phiên — hit-rate {hit*100:.0f}%, có ý nghĩa thống kê (p={p:.2f})"
    else:
        color = "#b7791f"
        headline = "CHƯA ĐỦ BẰNG CHỨNG THỐNG KÊ — chưa phân biệt được với dự báo ngẫu nhiên ở mọi khung thời gian"

    peer_txt = (
        f"gộp {len(result.peers_used)} mã cùng nhóm" if result.peers_used
        else "chỉ dùng lịch sử riêng mã này"
    )
    caption = (
        f"k={result.calibrated_k} · {peer_txt} · gộp {result.n_folds_used} giai đoạn walk-forward"
    )

    # dung ky tu tron dac "●" (khong phai emoji mau) lam dau hieu mau - de
    # dam bao render dung tren moi font/ung dung dan (Zalo/Messenger/Word...)
    # khong phu thuoc emoji-font co san hay khong nhu 🟢/🔴/🟡.
    return (
        f"<span style='color:{color};font-size:13pt;'>●</span> "
        f"<b style='color:{color};font-size:11.5pt;'>{_html.escape(headline)}</b><br>"
        f"<span style='color:#888888;font-size:8.5pt;'>Chi tiết từng khung thời gian xem bảng số ở trên. "
        f"{_html.escape(caption)}.</span>"
    )


_SIGNAL_SHORT_NAME = {
    "Xu hướng EMA": "EMA",
    "RSI(14)": "RSI",
    "MACD": "MACD",
    "ADX(14) - sức mạnh xu hướng": "ADX",
    "Khối lượng": "Volume",
    "Mây Ichimoku (Kumo)": "Kumo",
    "Tenkan/Kijun": "Tenkan/Kijun",
}


def _short_signal_tag(signal) -> str:
    """Rut gon 1 Signal (classic_ta.py) thanh 1 the ngan (vd 'RSI=30',
    'Hammer') - bo cau van day dong dai, chi giu ten + so lieu quan trong
    nhat (neu co dang 'TEN=SO' trong detail)."""
    name = signal.name
    if name.startswith("Mô hình nến:"):
        short = name.split(":", 1)[1].split("(")[0].strip()
    else:
        short = _SIGNAL_SHORT_NAME.get(name, name.split("(")[0].strip())
    m = re.search(r"[A-Za-z]+=\d+", signal.detail)
    if m and m.group(0).split("=")[0] not in short:
        short = f"{short} {m.group(0)}"
    return short


def _classic_verdict_explanation(scorecard) -> str:
    """Giai thich chi tiet hon y nghia cua verdict - theo phan hoi nguoi dung
    2026-08-03 rang phan 'Trung lập/hỗn hợp' can duoc giai thich ky hon (truoc
    day chi hien verdict + the ngan, khong noi ro 'trung lap' nghia la gi va
    nen lam gi tiep). Danh rieng 1 doan giai thich dai hon, de doc hon cho
    truong hop trung lap/hon hop - vi day la truong hop de gay hieu lam nhat
    (khong ro la "tin hieu yeu" hay "chua co du lieu" hay "thi truong dung
    yen") so voi tich cuc/tieu cuc von da tuong doi tu giai thich."""
    if scorecard.verdict == "Tích cực":
        return f"Phần lớn tín hiệu kỹ thuật đang nghiêng về hướng TĂNG ({scorecard.verdict_detail})."
    if scorecard.verdict == "Tiêu cực":
        return f"Phần lớn tín hiệu kỹ thuật đang nghiêng về hướng GIẢM ({scorecard.verdict_detail})."
    # "Trung lập" (khong du tin hieu duoc cham) hoac "Trung lập/hỗn hợp" (tin hieu tang/giam gan can bang)
    return (
        f"Các tín hiệu kỹ thuật đang TRÁI CHIỀU NHAU — {scorecard.verdict_detail}, nên CHƯA "
        f"nghiêng rõ về hướng tăng hay giảm. Đây không hẳn là dấu hiệu xấu, mà thường có nghĩa "
        f"cổ phiếu đang trong giai đoạn giằng co/tích lũy, hoặc các chỉ báo ngắn hạn và trung "
        f"hạn đang mâu thuẫn nhau. Nên cân nhắc CHỜ THÊM tín hiệu đồng thuận (nhiều chỉ báo cùng "
        f"chiều rõ ràng hơn) trước khi ra quyết định, thay vì hành động dựa trên đánh giá này."
    )


def _scorecard_to_html(scorecard) -> str:
    """Tom tat ClassicScorecard (classic_ta.py) thanh 1 dong verdict + cac THE
    ngan, cong 1 doan giai thich ro hon ben duoi (them 2026-08-03 theo phan
    hoi nguoi dung: truong hop 'Trung lập/hỗn hợp' can giai thich chi tiet
    hon, dung font to hon va mau de doc hon). Danh sach chi tiet day du van
    con trong scorecard.signals[*].detail neu can dung lai sau nay (vd
    tooltip)."""
    # Mau amber cu (#8a6d00) qua toi/kho doc tren nen trang - doi sang mau
    # cam-vang dam hon, tuong phan tot hon (dua theo bang mau Chakra UI
    # orange.600, thiet ke rieng cho van ban canh bao tren nen trang).
    verdict_color = {"Tích cực": "#2e7d32", "Tiêu cực": "#c62828"}.get(scorecard.verdict, "#b7791f")
    tag_color = {"bullish": "#2e7d32", "bearish": "#c62828", "neutral": "#757575"}
    arrow = {"bullish": "▲", "bearish": "▼", "neutral": "•"}
    tags = [
        f"<span style='color:{tag_color[s.direction]};'>{arrow[s.direction]} {_html.escape(_short_signal_tag(s))}</span>"
        for s in scorecard.signals
    ]
    tag_line = "&nbsp;&nbsp;·&nbsp;&nbsp;".join(tags)
    explanation = _classic_verdict_explanation(scorecard)
    return (
        f"<b style='color:{verdict_color};font-size:13pt;'>{_html.escape(scorecard.verdict)}</b>"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;{tag_line}<br>"
        f"<span style='color:#3a3a3a;font-size:10pt;'>{_html.escape(explanation)}</span><br>"
        f"<span style='color:#999999;font-size:8pt;'>{_html.escape(scorecard.disclaimer)}</span>"
    )


class WorkerThread(QThread):
    """Chay fetch/update trong thread rieng de khong lam dong UI (network co the cham)."""

    finished_ok = pyqtSignal(object)
    finished_err = pyqtSignal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args

    def run(self):
        try:
            result = self.fn(*self.args)
            self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.finished_err.emit(str(e))


class ScreenerWorker(QThread):
    """Quet TOAN BO danh sach ma (Tab 3 - xem market_screener.py) trong 1
    thread rieng, khac WorkerThread thong thuong o cho: (1) day la vong lap
    QUA NHIEU ma, khong phai 1 lan goi ham duy nhat, (2) phat tin hieu KET
    QUA TUNG MA ngay khi xong (row_ready) thay vi doi het toan bo moi tra ve
    - can thiet vi quet toan thi truong co the mat rat lau (xem canh bao
    trong market_screener.py), nguoi dung can thay ket qua dan dan, va (3)
    ho tro DUNG GIUA CHUNG (stop()) ma khong mat cac ma da quet xong."""

    progress = pyqtSignal(int, int, str)   # (da xong, tong so, ma dang quet)
    row_ready = pyqtSignal(object)          # market_screener.ScreenerRow
    finished_all = pyqtSignal()

    def __init__(self, conn, symbols: list[str]):
        super().__init__()
        self.conn = conn
        self.symbols = symbols
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        total = len(self.symbols)
        for i, symbol in enumerate(self.symbols, start=1):
            if self._stop_requested:
                break
            self.progress.emit(i, total, symbol)
            try:
                row = screener.screen_one_symbol(self.conn, symbol)
            except Exception as e:  # noqa: BLE001
                row = screener.ScreenerRow(symbol, "error", message=str(e))
            self.row_ready.emit(row)
        self.finished_all.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phân tích cổ phiếu Việt Nam")
        self.resize(1000, 700)

        self.conn = dm.get_connection(DB_PATH)
        self._worker: WorkerThread | None = None
        self._analysis_worker: WorkerThread | None = None

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        data_tab = QWidget()
        tabs.addTab(data_tab, "Dữ liệu")
        layout = QVBoxLayout(data_tab)

        # --- input row ---
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Mã cổ phiếu:"))
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("VD: FPT, VNM, HPG...")
        self.ticker_input.setMaximumWidth(150)
        self.ticker_input.returnPressed.connect(self.on_update_clicked)
        input_row.addWidget(self.ticker_input)

        self.btn_update = QPushButton("Cập nhật dữ liệu")
        self.btn_update.clicked.connect(self.on_update_clicked)
        input_row.addWidget(self.btn_update)

        self.btn_refresh = QPushButton("Kéo lại toàn bộ (chia cổ tức)")
        self.btn_refresh.clicked.connect(self.on_full_refresh_clicked)
        input_row.addWidget(self.btn_refresh)

        self.btn_load = QPushButton("Xem dữ liệu đã lưu")
        self.btn_load.clicked.connect(self.on_load_clicked)
        input_row.addWidget(self.btn_load)

        input_row.addStretch()
        layout.addLayout(input_row)

        # --- status log ---
        layout.addWidget(QLabel("Nhật ký:"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        layout.addWidget(self.log)

        # --- preview table ---
        layout.addWidget(QLabel("Dữ liệu gần nhất (20 dòng cuối trong DB):"))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Ngày", "Mở", "Cao", "Thấp", "Đóng", "Khối lượng"])
        layout.addWidget(self.table)

        self._log(f"DB: {DB_PATH}")

        # ================= TAB 2: Phan tich ky thuat (Phase 2-4) =================
        # Layout dang "the bao cao" (report card) theo yeu cau nguoi dung
        # 2026-07-29: de nguoi dung co the chup man hinh hoac dung nut
        # "Sao chep bao cao (anh)" gui cho ban be danh gia, toan bo ket qua
        # (tieu de/subtitle/bang so/chart/nhan dinh/scorecard) nam trong 1
        # QFrame trang, bo goc, vien mong - giong 1 khoi bao cao lien mach
        # thay vi cac widget roi rac nhu truoc.
        analysis_tab = QWidget()
        tabs.addTab(analysis_tab, "Phân tích kỹ thuật")
        a_outer_layout = QVBoxLayout(analysis_tab)

        a_input_row = QHBoxLayout()
        a_input_row.addWidget(QLabel("Mã cổ phiếu:"))
        self.analysis_ticker_input = QLineEdit()
        self.analysis_ticker_input.setPlaceholderText("VD: FPT, VNM, HPG...")
        self.analysis_ticker_input.setMaximumWidth(150)
        a_input_row.addWidget(self.analysis_ticker_input)

        self.btn_analyze = QPushButton("Chạy phân tích")
        self.btn_analyze.clicked.connect(self.on_analyze_clicked)
        a_input_row.addWidget(self.btn_analyze)

        self.btn_copy_report = QPushButton("📋 Sao chép báo cáo (ảnh)")
        self.btn_copy_report.setEnabled(False)
        self.btn_copy_report.setToolTip(
            "Chụp toàn bộ khung báo cáo (tiêu đề, bảng số, biểu đồ, nhận định) "
            "vào clipboard - dán bằng Ctrl+V vào Zalo/Messenger/Word... để gửi."
        )
        self.btn_copy_report.clicked.connect(self.on_copy_report_clicked)
        a_input_row.addWidget(self.btn_copy_report)

        a_input_row.addStretch()
        a_outer_layout.addLayout(a_input_row)

        a_outer_layout.addWidget(QLabel(
            "Lưu ý: cần có dữ liệu của mã này VÀ của VNINDEX trong DB (dùng tab "
            "'Dữ liệu' để cập nhật trước, hoặc để app tự động tải VNINDEX nếu chưa có)."
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        a_outer_layout.addWidget(scroll, stretch=1)

        self.report_frame = QFrame()
        self.report_frame.setObjectName("reportFrame")
        # tang tu 760 -> 900 (2026-08-03): metrics_table them 3 cot gia +
        # 1 dong horizon 50, can rong hon de khong bi chat chu tieu de cot.
        self.report_frame.setMinimumWidth(900)
        self.report_frame.setStyleSheet(
            "QFrame#reportFrame { background-color: #ffffff; border: 1px solid #dddddd; "
            "border-radius: 10px; }"
            "QLabel#reportTitle { font-size: 17pt; font-weight: 600; color: #1a1a1a; }"
            "QLabel#reportSubtitle { font-size: 9.5pt; color: #666666; }"
            "QLabel#sectionHeader { font-size: 10pt; font-weight: 600; color: #888888; "
            "margin-top: 4px; }"
            "QLabel#verdictBox { background-color: #f7f7f7; border: 1px solid #e5e5e5; "
            "border-radius: 6px; padding: 8px 10px; }"
        )
        scroll.setWidget(self.report_frame)

        report_layout = QVBoxLayout(self.report_frame)
        report_layout.setContentsMargins(22, 18, 22, 18)
        report_layout.setSpacing(8)

        self.report_title = QLabel("Chưa có kết quả - nhập mã cổ phiếu và bấm 'Chạy phân tích'.")
        self.report_title.setObjectName("reportTitle")
        self.report_title.setWordWrap(True)
        report_layout.addWidget(self.report_title)

        self.report_subtitle = QLabel("")
        self.report_subtitle.setObjectName("reportSubtitle")
        self.report_subtitle.setWordWrap(True)
        report_layout.addWidget(self.report_subtitle)

        # --- Tom tat 1 dong (TL;DR) - hien ngay duoi subtitle, TRUOC bang so
        # va chart, de nguoi xem nam duoc ket luan chinh ngay khong can doc
        # het bao cao (theo phan hoi nguoi dung: "nhieu chu, khong tom gon"). ---
        self.rationale_label = QLabel("")
        self.rationale_label.setObjectName("verdictBox")
        self.rationale_label.setTextFormat(Qt.TextFormat.RichText)
        self.rationale_label.setWordWrap(True)
        report_layout.addWidget(self.rationale_label)

        self.classic_label = QLabel("")
        self.classic_label.setObjectName("verdictBox")
        self.classic_label.setTextFormat(Qt.TextFormat.RichText)
        self.classic_label.setWordWrap(True)
        report_layout.addWidget(self.classic_label)

        metrics_header = QLabel("Chi tiết theo khung thời gian (3 / 10 / 20 / 50 phiên)")
        metrics_header.setObjectName("sectionHeader")
        report_layout.addWidget(metrics_header)

        # Them 2026-08-03 theo yeu cau nguoi dung: "bổ sung thêm giá tại phiên
        # 3,10,20" - analysis_engine.generate_assessment DA TINH SAN gia
        # tuyet doi (price_path/lower_price_path/upper_price_path, xem
        # forecasts[h] trong analysis_engine.py) tu truoc, chi la chua duoc
        # hien ra UI (chi hien % thay doi). Them 3 cot gia canh 3 cot % tuong
        # ung de nguoi dung khong phai tu quy doi % ra VND. Cung dip nay them
        # horizon 50 phien (analysis_engine.DEFAULT_HORIZONS) nen bang co the
        # co toi 4 dong (3/10/20/50) tuy horizons cua ket qua tra ve.
        self.metrics_table = QTableWidget(0, 11)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Phiên", "Kỳ vọng %", "Giá kỳ vọng", "Dưới %", "Giá dưới", "Trên %", "Giá trên",
             "Tin cậy OOT", "Baseline", "p-value", "Số mẫu"]
        )
        self.metrics_table.setToolTip(
            "Giá kỳ vọng/dưới/trên = giá tuyệt đối (VND) suy ra từ % kỳ vọng/dưới/trên áp vào giá hiện "
            "tại - cùng thông tin với cột %, chỉ đổi đơn vị cho dễ đọc. "
            "Tin cậy OOT = hit-rate hướng đi, gộp qua nhiều giai đoạn walk-forward (kiểm tra ngoài mẫu). "
            "Baseline = hit-rate của dự báo ngây thơ (không xét trạng thái hiện tại). "
            "p-value = kiểm định nhị thức so với giả thuyết 'không có tín hiệu' (50%) - "
            "p<0.10 mới coi là có ý nghĩa thống kê."
        )
        self.metrics_table.setMaximumHeight(170)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        report_layout.addWidget(self.metrics_table)

        # --- chart: nen Nhat + gia lich su + duong du bao nhieu doan (Buoc 4) ---
        self.figure = Figure(figsize=(8, 4.5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(380)
        self.chart_ax = self.figure.add_subplot(111)
        report_layout.addWidget(self.canvas)

        report_layout.addStretch()

        # ================= TAB 3: Quet toan thi truong (2026-08-03) =================
        # Theo yeu cau nguoi dung: "toàn những cổ phiếu tôi để ý ... tôi
        # muốn có thêm 1 tab ... list ra những cổ phiếu Thuần kỹ thuật là
        # tích cực hoặc đánh giá là đáng tin và tích cực" - khi hoi ro pham
        # vi danh sach, nguoi dung chon "lay toan bo co phieu dang niem yet
        # luon" (khong gioi han theo dang ky watchlist rieng). XEM CANH BAO
        # QUAN TRONG trong market_screener.py: quet toan thi truong la 1 tac
        # vu RAT NANG (co the mat hang chuc phut den vai gio, dac biet lan
        # dau), nen chay ngam + co the dung + hien ket qua dan dan tung ma.
        self._screener_worker: ScreenerWorker | None = None
        self._screener_rows: list["screener.ScreenerRow"] = []

        screener_tab = QWidget()
        tabs.addTab(screener_tab, "Quét toàn thị trường")
        sc_layout = QVBoxLayout(screener_tab)

        sc_info_label = QLabel(
            "Quét TOÀN BỘ mã cổ phiếu đang niêm yết (HOSE/HNX/UPCOM) và liệt kê mã 'Thuần kỹ thuật "
            "tích cực' (đánh giá định tính - CHƯA kiểm chứng thống kê) hoặc 'Đáng tin & tích cực' (có ý "
            "nghĩa thống kê qua walk-forward, hit-rate>50%, xem tab 'Phân tích kỹ thuật' để hiểu 2 khái "
            "niệm này). ⚠ Quét toàn thị trường có thể mất RẤT NHIỀU thời gian (hàng chục phút đến vài "
            "giờ ở lần quét đầu, do phải tải dữ liệu cho hàng nghìn mã) - có thể bấm 'Dừng' bất cứ lúc "
            "nào, các mã đã quét xong vẫn giữ nguyên kết quả."
        )
        sc_info_label.setWordWrap(True)
        sc_layout.addWidget(sc_info_label)

        sc_input_row = QHBoxLayout()
        self.btn_screen_start = QPushButton("Bắt đầu quét toàn thị trường")
        self.btn_screen_start.clicked.connect(self.on_screen_start_clicked)
        sc_input_row.addWidget(self.btn_screen_start)

        self.btn_screen_stop = QPushButton("Dừng")
        self.btn_screen_stop.setEnabled(False)
        self.btn_screen_stop.clicked.connect(self.on_screen_stop_clicked)
        sc_input_row.addWidget(self.btn_screen_stop)

        self.btn_screen_export = QPushButton("💾 Xuất Excel")
        self.btn_screen_export.setToolTip(
            "Xuất CHÍNH XÁC các dòng đang hiển thị trong bảng kết quả bên dưới (đã áp dụng bộ lọc) ra file .xlsx."
        )
        self.btn_screen_export.clicked.connect(self.on_screen_export_clicked)
        sc_input_row.addWidget(self.btn_screen_export)

        sc_input_row.addStretch()
        sc_layout.addLayout(sc_input_row)

        # --- Bo loc theo verdict (them 2026-08-03 theo yeu cau nguoi dung:
        # "co filter trung lap, tich cuc, tieu cuc") - thay cho 1 checkbox
        # "chi hien tich cuc/dang tin" duy nhat truoc day (qua gop chung 2
        # khai niem "thuan ky thuat tich cuc" va "dang tin thong ke" lam 1).
        # Tach lam 3 nhom doc lap: (1) 3 checkbox theo verdict dinh tinh cua
        # classic_ta (mac dinh deu bat - hien tat ca), (2) 1 checkbox rieng
        # cho ma "chua danh gia" (status skip/error, khong co verdict - mac
        # dinh TAT, vi day khong phai ket qua phan tich thuc su); (3) 1
        # checkbox loc them theo "dang tin thong ke" (k-NN, mac dinh tat).
        # SUA 2026-08-03 (bao loi tu nguoi dung): truoc day ma "chua danh
        # gia" luon hien BAT KE 3 checkbox verdict o tren co duoc bat/tat gi
        # khong - vi verdict rong ("") khong khop bat ky 1 trong 3 chuoi
        # "Tích cực"/"Trung lập.../"Tiêu cực" nen khong bi checkbox nao loc
        # ra - dan den khi nguoi dung chi bat "Tich cuc" van thay ca loat
        # dong "—" (khong co du lieu) xen vao. Them checkbox rieng #4 de nguoi
        # dung tu quyet dinh co muon thay nhung ma nay khong. ---
        sc_filter_row = QHBoxLayout()
        sc_filter_row.addWidget(QLabel("Lọc theo đánh giá thuần kỹ thuật:"))

        self.chk_verdict_positive = QCheckBox("Tích cực")
        self.chk_verdict_neutral = QCheckBox("Trung lập")
        self.chk_verdict_negative = QCheckBox("Tiêu cực")
        for chk in (self.chk_verdict_positive, self.chk_verdict_neutral, self.chk_verdict_negative):
            chk.setChecked(True)
            chk.stateChanged.connect(self.on_screen_filter_changed)
            sc_filter_row.addWidget(chk)

        self.chk_verdict_unrated = QCheckBox("Chưa đánh giá (lỗi/thiếu dữ liệu)")
        self.chk_verdict_unrated.setToolTip(
            "Mã chưa chạy được đánh giá thuần kỹ thuật (thiếu dữ liệu, không tải được, hoặc lỗi khác) - "
            "xem cột 'Ghi chú' để biết lý do cụ thể."
        )
        self.chk_verdict_unrated.setChecked(False)
        self.chk_verdict_unrated.stateChanged.connect(self.on_screen_filter_changed)
        sc_filter_row.addWidget(self.chk_verdict_unrated)

        self.chk_screen_stat_only = QCheckBox("Chỉ hiện mã đáng tin thống kê (k-NN)")
        self.chk_screen_stat_only.setToolTip(
            "Chỉ hiện mã có ít nhất 1 khung thời gian với hit-rate OOT>50% VÀ có ý nghĩa "
            "thống kê (p<0.10) - xem tooltip cột 'Đáng tin & tích cực'."
        )
        self.chk_screen_stat_only.stateChanged.connect(self.on_screen_filter_changed)
        sc_filter_row.addWidget(self.chk_screen_stat_only)

        sc_filter_row.addStretch()
        sc_layout.addLayout(sc_filter_row)

        self.screen_progress_label = QLabel("Chưa quét lần nào.")
        self.screen_progress_label.setWordWrap(True)
        sc_layout.addWidget(self.screen_progress_label)

        self.screen_table = QTableWidget(0, 4)
        self.screen_table.setHorizontalHeaderLabels(["Mã", "Thuần kỹ thuật", "Đáng tin & tích cực", "Ghi chú"])
        self.screen_table.setToolTip(
            "Thuần kỹ thuật = đánh giá định tính (EMA/RSI/MACD/ADX/volume/Ichimoku/mô hình nến), CHƯA "
            "kiểm chứng thống kê. Đáng tin & tích cực = có ít nhất 1 khung thời gian (3/10/20/50 phiên) mà "
            "hit-rate ngoài mẫu (OOT) > 50% VÀ có ý nghĩa thống kê (p<0.10) qua walk-forward - CHỈ được "
            "tính cho mã đã 'Thuần kỹ thuật: Tích cực' trước (để giảm tải khi quét hàng nghìn mã)."
        )
        self.screen_table.setAlternatingRowColors(True)
        self.screen_table.verticalHeader().setVisible(False)
        self.screen_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        sc_layout.addWidget(self.screen_table)

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    def _set_busy(self, busy: bool) -> None:
        self.btn_update.setEnabled(not busy)
        self.btn_refresh.setEnabled(not busy)
        self.btn_load.setEnabled(not busy)
        self.ticker_input.setEnabled(not busy)

    def _get_symbol(self) -> str | None:
        symbol = self.ticker_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Thiếu mã cổ phiếu", "Vui lòng nhập mã cổ phiếu.")
            return None
        return symbol

    # ------------------------------------------------------------------
    def on_update_clicked(self) -> None:
        symbol = self._get_symbol()
        if not symbol:
            return
        self._log(f"Đang cập nhật dữ liệu cho {symbol}...")
        self._set_busy(True)
        self._worker = WorkerThread(dm.update_stock_data, self.conn, symbol)
        self._worker.finished_ok.connect(self._on_update_done)
        self._worker.finished_err.connect(self._on_worker_error)
        self._worker.start()

    def on_full_refresh_clicked(self) -> None:
        symbol = self._get_symbol()
        if not symbol:
            return
        confirm = QMessageBox.question(
            self,
            "Xác nhận",
            f"Kéo lại TOÀN BỘ dữ liệu cho {symbol}? Dữ liệu cũ sẽ bị xóa và tải lại từ đầu.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._log(f"Đang kéo lại toàn bộ dữ liệu cho {symbol}...")
        self._set_busy(True)
        self._worker = WorkerThread(dm.full_refresh, self.conn, symbol)
        self._worker.finished_ok.connect(self._on_update_done)
        self._worker.finished_err.connect(self._on_worker_error)
        self._worker.start()

    def on_load_clicked(self) -> None:
        symbol = self._get_symbol()
        if not symbol:
            return
        self._render_preview(symbol)

    def _on_update_done(self, result: dm.UpdateResult) -> None:
        self._set_busy(False)
        self._log(result.message)
        if result.action == "error":
            QMessageBox.warning(self, "Lỗi", result.message)
        self._render_preview(result.symbol)

    def _on_worker_error(self, err: str) -> None:
        self._set_busy(False)
        self._log(f"LỖI: {err}")
        QMessageBox.critical(self, "Lỗi không mong đợi", err)

    def _render_preview(self, symbol: str) -> None:
        df = dm.get_price_df(self.conn, symbol)
        if df.empty:
            self._log(f"Chưa có dữ liệu cho {symbol} trong DB.")
            self.table.setRowCount(0)
            return
        tail = df.tail(20)
        self.table.setRowCount(len(tail))
        for row_idx, (date, row) in enumerate(tail.iterrows()):
            values = [
                date.strftime("%Y-%m-%d"),
                f"{row['open']:.2f}",
                f"{row['high']:.2f}",
                f"{row['low']:.2f}",
                f"{row['close']:.2f}",
                f"{int(row['volume'])}",
            ]
            for col_idx, val in enumerate(values):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(val))

    # ------------------------------------------------------------------
    # Tab 2: Phan tich ky thuat
    # ------------------------------------------------------------------
    def on_analyze_clicked(self) -> None:
        symbol = self.analysis_ticker_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Thiếu mã cổ phiếu", "Vui lòng nhập mã cổ phiếu.")
            return
        self.report_title.setText(f"Đang phân tích {symbol}...")
        self.report_subtitle.setText("(có thể mất vài giây do phải tính k-NN + calibrate)")
        self.rationale_label.setText("")
        self.classic_label.setText("")
        self.btn_copy_report.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self._analysis_worker = WorkerThread(_run_full_analysis, self.conn, symbol)
        self._analysis_worker.finished_ok.connect(self._on_analysis_done)
        self._analysis_worker.finished_err.connect(self._on_analysis_error)
        self._analysis_worker.start()

    def on_copy_report_clicked(self) -> None:
        """Chup toan bo khung bao cao (tieu de + bang so + chart + nhan dinh)
        thanh 1 anh va dua vao clipboard - theo yeu cau nguoi dung de co the
        dan (Ctrl+V) gui cho ban be danh gia, khong can chup man hinh thu cong."""
        pixmap = self.report_frame.grab()
        QGuiApplication.clipboard().setPixmap(pixmap)
        self._log("Đã sao chép báo cáo (hình ảnh) vào clipboard - dán bằng Ctrl+V để gửi.")

    def _on_analysis_done(self, bundle: "FullAnalysisBundle") -> None:
        self.btn_analyze.setEnabled(True)
        self._render_analysis_result(bundle)

    def _on_analysis_error(self, err: str) -> None:
        self.btn_analyze.setEnabled(True)
        self.report_title.setText("Lỗi phân tích")
        self.report_subtitle.setText(err)
        QMessageBox.critical(self, "Lỗi phân tích", err)

    def _render_analysis_result(self, bundle: "FullAnalysisBundle") -> None:
        result = bundle.assessment
        scorecard = bundle.classic_scorecard
        horizons = list(result.forecasts.keys())
        self.metrics_table.setRowCount(len(horizons))
        for row_idx, h in enumerate(horizons):
            f = result.forecasts[h]
            m = result.oot_metrics.get(h)
            b = result.baseline_oot_metrics.get(h)
            p = result.p_values.get(h)
            hit_txt = f"{m.hit_rate * 100:.0f}%" if (m and m.n_samples > 0) else "n/a"
            baseline_txt = f"{b.hit_rate * 100:.0f}%" if (b and b.n_samples > 0) else "n/a"
            if p is not None and m and m.n_samples > 0:
                p_txt = f"{p:.2f}" + ("*" if p < ae.SIGNIFICANCE_ALPHA else "")
            else:
                p_txt = "n/a"
            n_txt = str(m.n_samples) if m else "0"
            # gia tuyet doi (VND) tai moc h - lay diem CUOI cua price_path/
            # lower_price_path/upper_price_path (danh sach nhieu doan tu 1..h,
            # da tinh san trong analysis_engine.generate_assessment).
            exp_price_txt = f"{f['price_path'][-1]:.2f}"
            lower_price_txt = f"{f['lower_price_path'][-1]:.2f}"
            upper_price_txt = f"{f['upper_price_path'][-1]:.2f}"
            values = [
                str(h),
                f"{f['expected_return_pct']:+.1f}",
                exp_price_txt,
                f"{f['lower_pct']:+.1f}",
                lower_price_txt,
                f"{f['upper_pct']:+.1f}",
                upper_price_txt,
                hit_txt,
                baseline_txt,
                p_txt,
                n_txt,
            ]
            for col_idx, val in enumerate(values):
                self.metrics_table.setItem(row_idx, col_idx, QTableWidgetItem(val))

        self.report_title.setText(f"{result.symbol} — Phân tích kỹ thuật")
        subtitle = f"Tính đến {result.as_of_date} · Giá hiện tại: {result.current_price:.2f}"
        if result.warnings:
            subtitle += " · ⚠ " + "; ".join(result.warnings)
        self.report_subtitle.setText(subtitle)

        self.rationale_label.setText(_summarize_oot_html(result))
        self.classic_label.setText(_scorecard_to_html(scorecard))

        # --- Buoc 4: ve chart nen Nhat + gia lich su + duong du bao nhieu doan ---
        history_df = dm.get_price_df(self.conn, result.symbol)
        if not history_df.empty:
            cv.plot_analysis(self.chart_ax, result.symbol, history_df, result)
            self.canvas.draw()

        self.btn_copy_report.setEnabled(True)

    # ------------------------------------------------------------------
    # Tab 3: Quet toan thi truong
    # ------------------------------------------------------------------
    def on_screen_start_clicked(self) -> None:
        self.screen_table.setRowCount(0)
        self._screener_rows = []
        self.btn_screen_start.setEnabled(False)
        self.btn_screen_stop.setEnabled(True)
        self.screen_progress_label.setText("Đang lấy danh sách mã đang niêm yết...")
        QApplication.processEvents()  # cho label cap nhat ngay truoc khi block 1 chut de goi API

        symbols = screener.fetch_all_listed_symbols()
        if not symbols:
            symbols = screener.FALLBACK_SYMBOLS
            self.screen_progress_label.setText(
                f"⚠ Không lấy được danh sách đầy đủ (đã thử VCI và VNDIRECT, có thể mạng lỗi/API đã đổi) - "
                f"dùng danh sách dự phòng ({len(symbols)} mã, KHÔNG đầy đủ toàn thị trường)."
            )
            self._log(
                "Quét toàn thị trường: không lấy được danh sách mã niêm yết đầy đủ, "
                f"dùng {len(symbols)} mã dự phòng."
            )

        self._screener_worker = ScreenerWorker(self.conn, symbols)
        self._screener_worker.progress.connect(self._on_screen_progress)
        self._screener_worker.row_ready.connect(self._on_screen_row_ready)
        self._screener_worker.finished_all.connect(self._on_screen_finished)
        self._screener_worker.start()

    def on_screen_stop_clicked(self) -> None:
        if self._screener_worker:
            self._screener_worker.stop()
        self.btn_screen_stop.setEnabled(False)
        self.screen_progress_label.setText(self.screen_progress_label.text() + " (đang dừng sau mã hiện tại...)")

    def on_screen_export_clicked(self) -> None:
        """Xuat ra file .xlsx - them 2026-08-03 theo yeu cau nguoi dung
        "export sang excel kết quả đang show trên bảng kết quả này". Doc
        TRUC TIEP tu self.screen_table (khong tu self._screener_rows) de
        dam bao xuat DUNG nhung gi nguoi dung dang thay tren man hinh (da
        qua bo loc verdict/dang tin), tranh lech giua man hinh va file xuat
        ra."""
        n_rows = self.screen_table.rowCount()
        if n_rows == 0:
            QMessageBox.information(
                self, "Không có dữ liệu", "Chưa có kết quả nào để xuất - hãy quét trước (hoặc bỏ bớt bộ lọc)."
            )
            return

        default_name = f"quet_thi_truong_{datetime.now():%Y%m%d_%H%M}.xlsx"
        path, _ = QFileDialog.getSaveFileName(self, "Xuất kết quả ra Excel", default_name, "Excel (*.xlsx)")
        if not path:
            return

        n_cols = self.screen_table.columnCount()
        headers = [self.screen_table.horizontalHeaderItem(c).text() for c in range(n_cols)]
        rows_data = []
        for r in range(n_rows):
            rows_data.append([
                (self.screen_table.item(r, c).text() if self.screen_table.item(r, c) else "")
                for c in range(n_cols)
            ])
        df = pd.DataFrame(rows_data, columns=headers)

        try:
            df.to_excel(path, index=False, engine="openpyxl")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi xuất Excel", f"Không xuất được file:\n{e}")
            return

        self._log(f"Đã xuất {n_rows} dòng kết quả (đang hiển thị) ra {path}")
        QMessageBox.information(self, "Xuất Excel thành công", f"Đã lưu {n_rows} dòng vào:\n{path}")

    def _on_screen_progress(self, done: int, total: int, symbol: str) -> None:
        self.screen_progress_label.setText(f"Đang quét {done}/{total}: {symbol}...")

    def _on_screen_row_ready(self, row: "screener.ScreenerRow") -> None:
        self._screener_rows.append(row)
        self._append_screen_row_if_visible(row)

    def _on_screen_finished(self) -> None:
        self.btn_screen_start.setEnabled(True)
        self.btn_screen_stop.setEnabled(False)
        n_classic = sum(1 for r in self._screener_rows if r.classic_positive)
        n_stat = sum(1 for r in self._screener_rows if r.stat_positive)
        self.screen_progress_label.setText(
            f"Đã quét xong {len(self._screener_rows)} mã — {n_classic} mã thuần kỹ thuật tích cực, "
            f"{n_stat} mã đáng tin & tích cực."
        )

    def on_screen_filter_changed(self) -> None:
        self.screen_table.setRowCount(0)
        for row in self._screener_rows:
            self._append_screen_row_if_visible(row)

    def _append_screen_row_if_visible(self, row: "screener.ScreenerRow") -> None:
        """Them 1 dong vao bang KET QUA - loc theo (1) 3 checkbox verdict
        (Tich cuc/Trung lap/Tieu cuc), (2) checkbox rieng cho ma "chua danh
        gia" (khong co verdict), va (3) checkbox 'dang tin thong ke' rieng.
        Goi lai tu on_screen_filter_changed khi bat/tat checkbox de ve lai
        toan bo tu self._screener_rows (nguon du lieu day du luon duoc giu,
        khong bi mat khi loc).

        SUA 2026-08-03 (bao loi tu nguoi dung, xem chi tiet o noi tao
        chk_verdict_unrated): ma verdict rong (status "skip"/"error") gio
        duoc xu ly nhu 1 nhanh RIENG, chiu su kiem soat cua
        chk_verdict_unrated - KHONG con tu dong "lot qua" ca 3 dieu kien
        Tich cuc/Trung lap/Tieu cuc nhu truoc (do khong khop chuoi nao ca)."""
        verdict = row.classic_verdict or ""  # rong neu ma bi "skip"/"error" (chua co danh gia)
        if not verdict:
            if not self.chk_verdict_unrated.isChecked():
                return
        elif verdict == "Tích cực" and not self.chk_verdict_positive.isChecked():
            return
        elif verdict.startswith("Trung lập") and not self.chk_verdict_neutral.isChecked():
            return
        elif verdict == "Tiêu cực" and not self.chk_verdict_negative.isChecked():
            return
        if self.chk_screen_stat_only.isChecked() and not row.stat_positive:
            return

        r = self.screen_table.rowCount()
        self.screen_table.insertRow(r)

        classic_txt = row.classic_verdict or "—"
        if row.stat_positive:
            stat_txt = (
                f"✓ {row.stat_best_horizon} phiên (hit {row.stat_best_hit_rate*100:.0f}%, "
                f"p={row.stat_best_pvalue:.2f})"
            )
        else:
            stat_txt = "—"
        note = row.message if row.status != "ok" else ""

        values = [row.symbol, classic_txt, stat_txt, note]
        for c, val in enumerate(values):
            item = QTableWidgetItem(val)
            if c == 1 and row.classic_positive:
                item.setForeground(QColor("#2e7d32"))
            if c == 2 and row.stat_positive:
                item.setForeground(QColor("#2e7d32"))
            if row.status == "error":
                item.setForeground(QColor("#c62828"))
            self.screen_table.setItem(r, c, item)


@dataclass
class FullAnalysisBundle:
    """Ket qua tong hop cho 1 lan 'Chay phan tich': ca phan du bao da kiem
    chung OOT (k-NN historical analogue, analysis_engine.py) VA phan danh
    gia dinh tinh (mo hinh nen + chi bao co dien, classic_ta.py) - 2 lop
    TACH BIET ro rang de nguoi dung khong nham lan muc do tin cay giua 2 ben
    (xem canh bao trong classic_ta.ClassicScorecard.disclaimer)."""
    assessment: ae.AssessmentResult
    classic_scorecard: "classic_ta.ClassicScorecard"


def _run_full_analysis(conn, symbol: str) -> FullAnalysisBundle:
    """Ham dieu phoi: dam bao du lieu la MOI NHAT co the (tu dong cap nhat,
    khong chi khi thieu), doc du lieu tu DB, roi goi
    analysis_engine.generate_assessment (du bao da kiem chung OOT) VA
    classic_ta.build_classic_scorecard (danh gia dinh tinh tu chi bao co dien
    + mo hinh nen candlestick_patterns.py). Ham nay o main.py (khong o
    analysis_engine.py) vi analysis_engine khong duoc phep phu thuoc vao
    DB/data_manager - giu tach biet trach nhiem.

    QUAN TRONG (sua 2026-08-03, theo phan hoi nguoi dung: bao cao van ghi
    "Tinh den 2026-07-29" du hom do da la 2026-08-03 - "thi truong thay doi
    lien tuc, dung du lieu out-of-date thi khong theo kip"): TRUOC DAY chi
    tu dong tai du lieu neu DB CHUA CO GI CA (dm.has_data) - nghia la ngay
    sau lan dau co du lieu, moi lan "Chay phan tich" tiep theo se KHONG BAO
    GIO tu dong lam moi du lieu nua, du da qua nhieu ngay/tuan - nguoi dung
    phai tu nho bam "Cap nhat du lieu" o tab 1 truoc moi lan phan tich. Bay
    gio: MOI LAN chay phan tich deu goi dm.update_stock_data (ham nay tu
    thong minh: full pull neu chua co, hoac xoa dong cuoi + tai lai tu do -
    incremental - neu da co) cho ca ma dang xet, VNINDEX, va tung peer gop
    vao. Ham update_stock_data KHONG BAO GIO nem exception (thay vao do tra
    ve UpdateResult.action="error" neu fetch that bai, vi du mat mang) nen
    goi truc tiep la an toan; neu that bai, van tiep tuc phan tich tren du
    lieu CU nhat dang co trong DB (khong lam nguoi dung bi chan hoan toan),
    nhung se them 1 canh bao ro rang vao ket qua de nguoi dung biet bao cao
    co the dang bi cu (thay vi im lang dung du lieu cu ma khong noi gi)."""
    fetch_warnings: list[str] = []

    vn_result = dm.update_stock_data(conn, "VNINDEX")
    if vn_result.action == "error":
        fetch_warnings.append(f"Không cập nhật được VNINDEX mới nhất ({vn_result.message}) - báo cáo có thể đang dùng dữ liệu cũ.")

    sym_result = dm.update_stock_data(conn, symbol)
    if sym_result.action == "error":
        fetch_warnings.append(f"Không cập nhật được {symbol} mới nhất ({sym_result.message}) - báo cáo có thể đang dùng dữ liệu cũ.")

    stock_df = dm.get_price_df(conn, symbol)
    vnindex_df = dm.get_price_df(conn, "VNINDEX")

    if stock_df.empty:
        raise ValueError(f"Không có dữ liệu cho {symbol} trong DB và không tải được từ VNDIRECT.")
    if vnindex_df.empty:
        raise ValueError("Không có dữ liệu VNINDEX trong DB và không tải được từ VNDIRECT.")
    if len(stock_df) < ae.MIN_USABLE_ROWS + ae.WARMUP_ROWS:
        raise ValueError(
            f"{symbol} chỉ có {len(stock_df)} dòng dữ liệu - cần ít nhất "
            f"~{ae.MIN_USABLE_ROWS + ae.WARMUP_ROWS} dòng (khoảng 2.5 năm giao dịch) "
            f"để chia Train/Val/OOT có ý nghĩa."
        )

    # --- gop hang xom cung nhom nganh/beta (xem sector_map.py) de tang co mau
    # cho danh gia thong ke - luon thu cap nhat MOI peer (best-effort), neu 1
    # peer bi loi tai/thieu du lieu thi bo qua peer do (khong lam hong ca lan
    # phan tich chinh), tu dong lui ve che do chi dung lich su rieng cua ma
    # neu KHONG peer nao dung duoc.
    peer_price_dfs = {}
    for peer_symbol in sector_map.get_peers(symbol):
        try:
            dm.update_stock_data(conn, peer_symbol)
            peer_df = dm.get_price_df(conn, peer_symbol)
            if not peer_df.empty:
                peer_price_dfs[peer_symbol] = peer_df
        except Exception:
            continue  # 1 peer loi khong duoc lam hong phan tich chinh

    assessment = ae.generate_assessment(symbol, stock_df, vnindex_df, peer_price_dfs=peer_price_dfs)
    if fetch_warnings:
        assessment.warnings = fetch_warnings + assessment.warnings

    # --- Phase 4: danh gia dinh tinh (chi bao co dien + mo hinh nen gan day) ---
    # TACH BIET voi k-NN OOT o tren - chi de tham khao nhanh, xem canh bao
    # trong classic_ta.ClassicScorecard.disclaimer.
    ind = ae.compute_indicators(stock_df)
    candle_hits = cp.recent_patterns(stock_df, n_recent=15)
    scorecard = classic_ta.build_classic_scorecard(ind.iloc[-1], candle_hits)

    return FullAnalysisBundle(assessment=assessment, classic_scorecard=scorecard)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
