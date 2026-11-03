"""
BÀI 01 — Cho AI theo dõi MỌI NGƯỜI trong video và lưu lại toàn bộ kết quả.

Chạy:   python 01_track_video.py
Kết quả: output/01_tracked_video.mp4  (mỗi người một màu + số ID)
         output/all_tracks.csv        (frame, id, x1, y1, x2, y2, conf — mọi khung của mọi ID)
         output/track_meta.json       (thông tin video + cài đặt đã dùng)

Nhắc lại từ yolo/04_tracking.py:
- Phát hiện: mỗi frame AI chỉ nói "ở đây có người".
- Theo dõi:  AI nói thêm "người này vẫn là người số 7 ở frame trước".

Tracker làm thế nào? Ở mỗi frame mới, nó:
  1. Đoán xem mỗi người đã biết sẽ đi tới đâu (dựa vào hướng và tốc độ chạy).
  2. Ghép từng khung mới với người đã biết có vị trí đoán gần nhất (khung chồng lên nhau nhiều nhất).
  3. Khung nào không ghép được với ai → cấp một ID MỚI.
  4. Người nào không thấy nữa → nhớ thêm một lúc (track_buffer), quá lâu thì quên.
Bài 05 ★ cho bạn tự viết một tracker đơn giản theo đúng ý tưởng này.

Khác bài yolo/04: bài này lưu CẢ KHUNG (x1, y1, x2, y2) và độ tin cậy, để các bài sau
chọn cầu thủ, vẽ lại video và phân tích mà KHÔNG phải chạy AI lại (chạy AI là phần chậm nhất).
"""

import csv
import time

import cv2
from ultralytics import YOLO

from config import (
    BASE_DIR,
    IMAGE_SIZE,
    META_JSON,
    MODEL_NAME,
    OUTPUT_DIR,
    PERSON_CLASS,
    TRACKS_CSV,
    VIDEO_PATH,
    require_video,
)
from tools import color_for_id, save_json

# 🔧 THỬ SỬA
CONFIDENCE_THRESHOLD = 0.3
MAX_FRAMES = 300           # chỉ xử lý 300 frame đầu cho nhanh; None = chạy hết video
SHOW_WINDOW = False        # True = xem trực tiếp khi đang chạy (bấm q để dừng)
# Chọn tracker:
#   "bytetrack.yaml"     — ByteTrack, nhanh (mặc định)
#   "botsort.yaml"       — BoT-SORT, bù được chuyển động camera, chậm hơn một chút
#   "my_bytetrack.yaml"  — bản ByteTrack bạn tự chỉnh (mở file đó ra xem)
TRACKER = "bytetrack.yaml"

require_video()

# File tracker tự chỉnh nằm trong thư mục này → đưa đường dẫn đầy đủ.
# Hai file có sẵn (bytetrack/botsort) thì thư viện tự biết tìm ở đâu.
tracker_path = str(BASE_DIR / TRACKER) if (BASE_DIR / TRACKER).exists() else TRACKER

# ------------------------------------------------------------------
# Bước 1: Nạp mô hình, mở video, chuẩn bị video đầu ra
# ------------------------------------------------------------------
model = YOLO(MODEL_NAME)

video = cv2.VideoCapture(str(VIDEO_PATH))
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {VIDEO_PATH.name}  —  {width}×{height}, {fps:.1f} fps, {total_frames} frame")
print(f"Tracker: {TRACKER}")

video_out_path = OUTPUT_DIR / "01_tracked_video.mp4"
writer = cv2.VideoWriter(str(video_out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# ------------------------------------------------------------------
# Bước 2: Vòng lặp — mỗi frame cho tracker xem, lưu mọi khung có ID
# ------------------------------------------------------------------
rows = []                  # mỗi phần tử: [frame, id, x1, y1, x2, y2, conf]
frame_index = 0
start_time = time.time()

while True:
    ok, frame = video.read()
    if not ok:
        break
    if MAX_FRAMES is not None and frame_index >= MAX_FRAMES:
        break

    # persist=True: "hãy nhớ những người ở frame trước" — thiếu nó thì mỗi frame lại đánh số từ đầu.
    results = model.track(frame, classes=[PERSON_CLASS], conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE,
                          persist=True, tracker=tracker_path, verbose=False)
    boxes = results[0].boxes

    # boxes.id là None khi tracker chưa gán ID cho ai (ví dụ frame không có người)
    if boxes.id is not None:
        for box, track_id in zip(boxes, boxes.id.tolist()):
            track_id = int(track_id)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            # Làm tròn 1 chữ số thập phân cho file CSV gọn
            rows.append([frame_index, track_id, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                         round(confidence, 3)])

            color = color_for_id(track_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"Frame {frame_index}  {frame_index / fps:.1f}s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    writer.write(frame)

    if SHOW_WINDOW:
        cv2.imshow("StatsLab - Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if frame_index % 30 == 0:
        print(f"  đang xử lý frame {frame_index}...")
    frame_index += 1

video.release()
writer.release()
cv2.destroyAllWindows()
elapsed = time.time() - start_time

if frame_index == 0:
    print("Không đọc được frame nào. Video có bị hỏng không?")
    raise SystemExit(1)

# ------------------------------------------------------------------
# Bước 3: Lưu kết quả
# ------------------------------------------------------------------
with open(TRACKS_CSV, "w", newline="", encoding="utf-8") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "id", "x1", "y1", "x2", "y2", "conf"])
    csv_writer.writerows(rows)

# Ghi lại cài đặt đã dùng. Sau này xem lại file kết quả, ta biết nó được tạo ra
# bằng mô hình nào, tracker nào, ngưỡng bao nhiêu (để chạy lại ra đúng như cũ).
meta = {
    "video": str(VIDEO_PATH),
    "fps": fps,
    "n_frames": total_frames,
    "frames_processed": frame_index,
    "width": width,
    "height": height,
    "stride": 1,
    "model": MODEL_NAME,
    "tracker": TRACKER,
    "conf": CONFIDENCE_THRESHOLD,
    "imgsz": IMAGE_SIZE,
}
save_json(meta, META_JSON)

# ------------------------------------------------------------------
# Bước 4: Tóm tắt
# ------------------------------------------------------------------
all_ids = {row[1] for row in rows}      # tập hợp (set): mỗi ID chỉ đếm một lần
print(f"\nĐã xử lý {frame_index} frame trong {elapsed:.1f} giây "
      f"(≈ {frame_index / elapsed:.1f} frame/giây trên máy của bạn)")
print(f"Tổng số khung đã lưu: {len(rows)}")
print(f"Tổng số ID khác nhau: {len(all_ids)}")
print(f"\n✅ Video kết quả: {video_out_path}")
print(f"✅ Mọi khung:     {TRACKS_CSV}")
print(f"✅ Cài đặt:       {META_JSON}")
print("\nTiếp theo: python 02_id_report.py  để xem các ID này 'sống' bao lâu.")
