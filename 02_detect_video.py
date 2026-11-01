"""
BÀI 02 — Chạy AI trên CẢ VIDEO, vẽ khung và đếm số người mỗi frame.

Chạy:   python 02_detect_video.py
Kết quả: output/02_detected_video.mp4   (video có khung quanh mọi người)
         output/02_people_count.csv     (mỗi dòng: frame, giây, số người)

Bạn sẽ học:
- Video = rất nhiều ảnh (frame) nối tiếp nhau. Chạy AI trên video
  thực chất là chạy AI trên TỪNG ẢNH một, trong một vòng lặp.
- fps (frames per second) = số ảnh mỗi giây. Frame số i ở giây thứ i / fps.
"""

import csv
import time

import cv2
from ultralytics import YOLO

from config import (
    IMAGE_SIZE,
    MODEL_NAME,
    OUTPUT_DIR,
    PERSON_CLASS,
    VIDEO_PATH,
    require_video,
)

# 🔧 THỬ SỬA các giá trị dưới đây
CONFIDENCE_THRESHOLD = 0.3
MAX_FRAMES = 300           # chỉ xử lý 300 frame đầu cho nhanh; đặt None để chạy hết video
SHOW_WINDOW = False        # True = mở cửa sổ xem trực tiếp khi đang chạy (bấm phím q để dừng)

require_video()

# ------------------------------------------------------------------
# Bước 1: Nạp mô hình và mở video
# ------------------------------------------------------------------
model = YOLO(MODEL_NAME)

video = cv2.VideoCapture(str(VIDEO_PATH))
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {VIDEO_PATH.name}")
print(f"  {width}×{height} pixel, {fps:.1f} fps, {total_frames} frame ≈ {total_frames / fps:.1f} giây")

# ------------------------------------------------------------------
# Bước 2: Chuẩn bị "máy ghi" video đầu ra (cùng kích thước, cùng fps)
# ------------------------------------------------------------------
video_out_path = OUTPUT_DIR / "02_detected_video.mp4"
codec = cv2.VideoWriter_fourcc(*"mp4v")          # kiểu nén video .mp4
writer = cv2.VideoWriter(str(video_out_path), codec, fps, (width, height))

# Danh sách để nhớ số người ở mỗi frame: mỗi phần tử là [frame, giây, số người]
counts = []

# ------------------------------------------------------------------
# Bước 3: Vòng lặp — đọc từng frame, cho AI xem, vẽ, ghi ra video
# ------------------------------------------------------------------
frame_index = 0
start_time = time.time()

while True:
    ok, frame = video.read()            # đọc frame tiếp theo
    if not ok:                          # hết video thì dừng
        break
    if MAX_FRAMES is not None and frame_index >= MAX_FRAMES:
        break

    results = model(frame, classes=[PERSON_CLASS], conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE, verbose=False)
    boxes = results[0].boxes
    num_people = len(boxes)

    # Vẽ từng khung
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Ghi thông tin lên góc trên-trái (không dấu vì putText không hỗ trợ tiếng Việt)
    seconds = frame_index / fps
    cv2.putText(frame, f"Frame {frame_index}  {seconds:.1f}s  {num_people} nguoi",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    writer.write(frame)
    counts.append([frame_index, round(seconds, 2), num_people])

    if SHOW_WINDOW:
        cv2.imshow("StatsLab - YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cứ 30 frame in tiến độ một lần để biết chương trình vẫn đang chạy
    if frame_index % 30 == 0:
        print(f"  đang xử lý frame {frame_index}... ({num_people} người)")
    frame_index += 1

video.release()
writer.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_time

# ------------------------------------------------------------------
# Bước 4: Lưu bảng đếm ra file CSV (mở được bằng Excel)
# ------------------------------------------------------------------
csv_path = OUTPUT_DIR / "02_people_count.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "seconds", "people"])     # dòng tiêu đề
    csv_writer.writerows(counts)

# ------------------------------------------------------------------
# Bước 5: Tóm tắt
# ------------------------------------------------------------------
if frame_index == 0:
    print("Không đọc được frame nào. Video có bị hỏng không?")
    raise SystemExit(1)

people_per_frame = [row[2] for row in counts]
print(f"\nĐã xử lý {frame_index} frame trong {elapsed:.1f} giây "
      f"(≈ {frame_index / elapsed:.1f} frame/giây trên máy của bạn)")
print(f"Số người mỗi frame: ít nhất {min(people_per_frame)}, "
      f"nhiều nhất {max(people_per_frame)}, "
      f"trung bình {sum(people_per_frame) / len(people_per_frame):.1f}")
print(f"\n✅ Video kết quả: {video_out_path}")
print(f"✅ Bảng đếm:      {csv_path}")
print("\nCâu hỏi: Trên sân có 10 cầu thủ (+ trọng tài). Vì sao số người AI đếm lại thay đổi qua từng frame?")
