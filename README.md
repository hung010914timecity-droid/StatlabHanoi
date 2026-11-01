# YOLO — code mẫu để thử nghiệm (StatsLab, tuần 2–3)

Thư mục này là code chạy trên **máy tính cá nhân** (Windows/Mac, không cần GPU), đi kèm buổi 4–5 và phần thử bóng của buổi 11 trong [STATSLAB_PLAN.md](../STATSLAB_PLAN.md). Mỗi file là một bài nhỏ, đọc từ trên xuống dưới. Tên biến và tên file viết bằng tiếng Anh; chú thích giải thích bằng tiếng Việt.

Mô hình dùng: **YOLO26** (`yolo26n.pt`), thế hệ YOLO mới nhất của Ultralytics.

## 1. Cài đặt (làm một lần)

Cần **Python 3.11 trở lên**. Mở Terminal (Mac) hoặc PowerShell (Windows), đi vào thư mục `yolo/`:

```bash
cd duong/dan/toi/statslab-project/yolo

python3 -m venv .venv                 # tạo "môi trường ảo" riêng cho dự án
source .venv/bin/activate             # Mac/Linux
# .venv\Scripts\activate              # Windows (dùng dòng này thay dòng trên)

pip install -r requirements.txt       # mất vài phút, tải khoảng 500 MB
```

Lần sau chỉ cần `cd` vào thư mục rồi chạy lại dòng `activate`.

## 2. Chuẩn bị video

Chép một clip bóng rổ **30–60 giây** (camera đứng yên, thấy toàn sân) vào `yolo/videos/` và đặt tên **`clip1.mp4`**.
Chưa có video? Bài 01 và 03 vẫn chạy được vì tự tải một ảnh mẫu từ Internet.

## 3. Các bài — chạy theo thứ tự

| Bài | Lệnh | Làm gì | Kết quả trong `output/` |
|---|---|---|---|
| 01 | `python 01_detect_image.py` | AI tìm người trong **một ảnh**; in tọa độ khung và độ tin cậy | `01_detected_image.jpg` |
| 02 | `python 02_detect_video.py` | Chạy AI trên **cả video**, đếm số người mỗi frame | `02_detected_video.mp4`, `02_people_count.csv` |
| 03 | `python 03_confidence_threshold.py` | So sánh 4 **ngưỡng tin cậy** trên cùng một ảnh | `03_threshold_compare.jpg` |
| 04 | `python 04_tracking.py` | **Tracking**: mỗi người một số ID; lưu vị trí chân | `04_tracked_video.mp4`, `04_positions.csv` |
| 05 ★ | `python 05_ball_detection.py` | Thử tìm **quả bóng**, xem AI thấy bóng ở bao nhiêu % frame | `05_ball_video.mp4` |

`config.py` chứa cài đặt dùng chung (đường dẫn video, tên mô hình, kích thước ảnh, thư mục). Không chạy file này trực tiếp.

Lần chạy đầu, thư viện tự tải file mô hình `yolo26n.pt` (~6 MB) vào thư mục đang đứng, nên cần có mạng.

## 4. Thử sửa gì?

Mỗi file có các dòng đánh dấu **🔧 THỬ SỬA** ở phía trên. Gợi ý:

- `CONFIDENCE_THRESHOLD`: thử 0.1 rồi 0.7. AI thấy nhiều hay ít người hơn? Có nhầm thêm không?
- `MAX_FRAMES`: đặt `None` để chạy hết video (lâu hơn).
- `SHOW_WINDOW = True`: xem video ngay lúc AI đang chạy (bấm `q` để dừng).
- Trong `config.py`, đổi `IMAGE_SIZE` thành 640 rồi chạy lại bài 01: AI bỏ sót bao nhiêu người ở xa? Nhanh hơn bao nhiêu?
- Trong `config.py`, đổi `MODEL_NAME` sang `"yolo26s.pt"`: mô hình lớn hơn chạy chậm hơn, nhưng có chính xác hơn không?
- Bài 05: đổi `IMAGE_SIZE` thành 1280 hoặc 640. AI tìm thấy bóng ít đi bao nhiêu?

## 5. Câu hỏi để thảo luận

1. Trên sân có 10 cầu thủ, vì sao bài 02 đếm được nhiều hơn 10 người, và con số lại thay đổi qua từng frame?
2. Ở bài 04, tổng số ID nhiều hơn số người trên sân rất nhiều. Đó là do đâu?
3. Ở bài 05, vì sao tìm bóng khó hơn tìm người rất nhiều?

## 6. Lỗi thường gặp

| Thông báo | Cách sửa |
|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` | Chưa bật môi trường ảo (`source .venv/bin/activate`) hoặc chưa `pip install -r requirements.txt` |
| `❌ Chưa có video` | Chép clip vào `yolo/videos/clip1.mp4` |
| Chạy rất chậm | Giảm `MAX_FRAMES`, giảm `IMAGE_SIZE` xuống 960 hoặc 640, giữ mô hình `yolo26n.pt` |
| Lỗi không tìm thấy `yolo26n.pt` | Bản `ultralytics` quá cũ: `pip install -U ultralytics` |
| Video kết quả không mở được | Mở bằng VLC (miễn phí) |
| Lỗi tải mô hình / ảnh mẫu | Kiểm tra kết nối Internet rồi chạy lại |
