# VN Stock Analysis

App phân tích kỹ thuật cổ phiếu Việt Nam - **1 thư mục dùng chung cho cả 2 bản**:

- **Bản Windows (desktop, PyQt6)**: `main.py` - chạy trực tiếp trên máy, dùng SQLite local. Không đổi gì so với trước.
- **Bản Web (deploy Vercel)**: `api/index.py` (FastAPI) + `index.html`/`analyze.html`/`screener.html` - dùng Postgres (Supabase/Neon). Hai bản dùng CHUNG các file phân tích (`analysis_engine.py`, `classic_ta.py`, `candlestick_patterns.py`, `sector_map.py`, `market_screener.py`, `chart_view.py`, `data_manager.py`) nằm ngay tại thư mục gốc - sửa 1 lần là cả 2 bản đều nhận thay đổi, không cần đồng bộ tay giữa 2 nơi.

## Cấu trúc

```
VN_StockApp/
├── main.py                  # App Windows (PyQt6) - Tab Dữ liệu / Phân tích / Quét thị trường
├── data_manager.py          # Lớp dữ liệu - DÙNG CHUNG (tự nhận SQLite hay Postgres)
├── analysis_engine.py       # Phân tích k-NN đã kiểm chứng OOT - DÙNG CHUNG
├── classic_ta.py            # Bảng điểm kỹ thuật cổ điển - DÙNG CHUNG
├── candlestick_patterns.py  # Nhận diện mô hình nến - DÙNG CHUNG
├── sector_map.py            # Nhóm ngành/peer - DÙNG CHUNG
├── market_screener.py       # Logic quét toàn thị trường - DÙNG CHUNG
├── chart_view.py            # Vẽ chart (dùng cho cả PyQt canvas và ảnh PNG trên web)
├── requirements.txt         # Dependency cho bản Windows (có PyQt6)
├── api/
│   ├── index.py             # FastAPI - toàn bộ route /api/* cho bản web
│   ├── screener_store.py    # Lưu tiến độ quét thị trường vào Postgres - CHỈ web dùng
│   └── requirements.txt     # Dependency CHO WEB (không có PyQt6)
├── index.html / analyze.html / screener.html / style.css / app.js   # Frontend tĩnh (web)
├── schema.sql                # Schema Postgres (tham khảo, API tự tạo bảng)
├── vercel.json                # Cấu hình Vercel Function + cron
└── .github/workflows/screener-cron.yml   # GitHub Actions gọi cron nhiều lần/giờ
```

## Chạy bản Windows

Như trước: `pip install -r requirements.txt` rồi `python main.py`.

## Deploy bản Web lên Vercel

### 1. Thiết lập Postgres (Supabase hoặc Neon)

Tạo project miễn phí tại [supabase.com](https://supabase.com) hoặc [neon.tech](https://neon.tech), lấy connection string dạng `postgres://user:password@host:port/dbname`. Không cần chạy `schema.sql` tay - API tự tạo bảng khi khởi động lần đầu.

### 2. Push lên GitHub và import vào Vercel

```powershell
cd G:\My Drive\03_Personal\StockInvestment\VN_StockApp
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/hungtrquoc/vnstock.git
git push -u origin main
```

Sau đó vào [vercel.com/new](https://vercel.com/new), import repo `hungtrquoc/vnstock`. Vercel sẽ nhận diện `api/index.py` là Python Function nhờ `vercel.json`.

### 3. Cấu hình biến môi trường trên Vercel

Project Settings → Environment Variables:
- `DATABASE_URL` = connection string Postgres ở bước 1.
- `CRON_SECRET` = một chuỗi bí mật tự đặt (ví dụ tạo bằng `openssl rand -hex 16`) - bảo vệ endpoint `/api/screener/cron`.

Deploy xong, mở `https://<project>.vercel.app/` để dùng.

### 4. Lưu ý về `requirements.txt`

Có **2 file `requirements.txt` riêng biệt** - đây là điểm dễ nhầm nhất khi gộp chung 1 thư mục: `requirements.txt` ở gốc là cho bản Windows (có PyQt6, không cài được/không cần trên Vercel), `api/requirements.txt` là cho bản web (fastapi/pandas/numpy/matplotlib/psycopg2-binary, không có PyQt6). Theo tài liệu Vercel, requirements.txt nằm cùng cấp với hàm (`api/`) được ưu tiên hơn requirements.txt ở gốc - nhưng **điều này chưa được kiểm chứng thực tế** (chưa có tài khoản Vercel để deploy thử). Nếu build trên Vercel báo lỗi liên quan PyQt6, đây chính là nguyên nhân cần xử lý tiếp.

## Quan trọng: giới hạn Cron của Vercel và cách vượt qua

Gói Vercel Hobby (miễn phí) chỉ cho phép cron job chạy **1 lần/ngày**. Quét toàn bộ ~1000-1700+ mã chia thành từng lô nhỏ (mặc định 30 mã/lần gọi `/api/screener/cron`) sẽ mất rất nhiều ngày nếu chỉ trông vào cron 1 lần/ngày của Vercel.

Cách khắc phục (đã cấu hình sẵn): `.github/workflows/screener-cron.yml` dùng GitHub Actions (miễn phí) gọi `/api/screener/cron` mỗi 10 phút - GitHub Actions chỉ đóng vai trò "chuông reo", toàn bộ tính toán vẫn chạy trên Vercel. Để bật:

1. Vào repo GitHub → Settings → Secrets and variables → Actions.
2. Thêm secret `VERCEL_APP_URL` (ví dụ `https://ten-project.vercel.app`) và `CRON_SECRET` (phải giống chính xác giá trị đã đặt trong Vercel).
3. Vào tab Actions → chọn workflow → "Run workflow" để chạy thử ngay, hoặc chờ chạy tự động theo lịch.

## Vì sao Tab "Quét toàn thị trường" trên web khác bản Windows

Bản Windows quét trực tiếp trong RAM khi bấm nút, hiện kết quả từng mã ngay khi xong. Vercel Function có giới hạn thời gian chạy (`maxDuration: 60` giây trong `vercel.json`), không thể quét hết ~1700 mã trong 1 lần gọi. Bản web tách thành 2 phần: `/api/screener/cron` chạy nền theo lô nhỏ, lưu kết quả + vị trí đã quét vào Postgres (`screener_results`/`screener_progress`); `screener.html` chỉ đọc kết quả đã lưu (`/api/screener/results`) và hiển thị thanh tiến độ, không phải "quét ngay khi bấm".

## Những phần chưa được kiểm chứng thực tế

- Chưa deploy thật lên Vercel (không có tài khoản Vercel trong môi trường phát triển) - cấu hình `vercel.json`/việc `api/requirements.txt` được ưu tiên đúng như mô tả ở trên vẫn cần xác nhận qua lần deploy thật.
- Nhánh Postgres của `data_manager.py`/`api/screener_store.py` được kiểm tra bằng một "shim" giả lập cú pháp psycopg2 trên SQLite (không có server Postgres thật để test) - nên chạy thử với `DATABASE_URL` Supabase/Neon thật trước khi tin tưởng hoàn toàn.
- Frontend (`*.html`) chưa được test bằng trình duyệt thật (chỉ kiểm tra cú pháp JavaScript và khớp tên trường JSON với API).
