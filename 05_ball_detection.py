"""
BÀI 05 (thử thách) — AI có tìm được QUẢ BÓNG không?

Chạy:   python 05_ball_detection.py
Kết quả: output/05_ball_video.mp4  (vòng tròn đỏ quanh quả bóng nếu AI thấy)

Mục đích KHÔNG phải để làm cho được, mà để thấy vì sao việc tự động
nhận diện "ném trúng / ném trượt" lại khó:
- Quả bóng NHỎ (chỉ vài chục pixel), bay NHANH (bị nhòe), hay bị TAY CHE.
- Mô hình có sẵn học trên ảnh đời thường, không chuyên về bóng rổ.
- Mô hình có sẵn KHÔNG biết "vành rổ" là gì (không có trong 80 loại vật thể).
Muốn tốt hơn phải tự huấn luyện lại mô hình trên ảnh sân bóng rổ — việc của giai đoạn sau.
"""

import cv2
from ultralytics import YOLO

from config import BALL_CLASS, MODEL_NAME, OUTPUT_DIR, VIDEO_PATH, require_video

# 🔧 THỬ SỬA: bóng khó thấy nên để ngưỡng thấp hơn so với tìm người
CONFIDENCE_THRESHOLD = 0.15
MAX_FRAMES = 300
# Bóng còn nhỏ hơn người nhiều, nên ở bài này dùng kích thước lớn hơn mức chung trong config.py.
# Ảnh lớn giúp AI thấy vật nhỏ rõ hơn, nhưng chạy chậm hơn (thử 640, 1280, 1920 rồi so sánh).
IMAGE_SIZE = 1920

require_video()

model = YOLO(MODEL_NAME)

video = cv2.VideoCapture(str(VIDEO_PATH))
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_out_path = OUTPUT_DIR / "05_ball_video.mp4"
writer = cv2.VideoWriter(str(video_out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_index = 0
frames_with_ball = 0

while True:
    ok, frame = video.read()
    if not ok:
        break
    if MAX_FRAMES is not None and frame_index >= MAX_FRAMES:
        break

    results = model(frame, classes=[BALL_CLASS], conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE, verbose=False)
    boxes = results[0].boxes

    if len(boxes) > 0:
        frames_with_ball += 1
        # Nếu AI thấy nhiều "quả bóng", chỉ lấy cái nó tin chắc nhất (khung đầu tiên)
        x1, y1, x2, y2 = [int(v) for v in boxes[0].xyxy[0].tolist()]
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = max(x2 - x1, y2 - y1) // 2 + 8
        cv2.circle(frame, center, radius, (0, 0, 255), 3)
        cv2.putText(frame, f"bong {float(boxes[0].conf[0]):.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    percent = 100 * frames_with_ball / (frame_index + 1)
    cv2.putText(frame, f"Frame {frame_index}  thay bong: {percent:.0f}% so frame",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    writer.write(frame)

    if frame_index % 30 == 0:
        print(f"  đang xử lý frame {frame_index}...")
    frame_index += 1

video.release()
writer.release()

if frame_index == 0:
    print("Không đọc được frame nào.")
    raise SystemExit(1)

print(f"\nAI thấy bóng ở {frames_with_ball}/{frame_index} frame "
      f"= {100 * frames_with_ball / frame_index:.0f}%")
print(f"✅ Video kết quả: {video_out_path}")
print("\nCâu hỏi: Xem video, những lúc AI KHÔNG thấy bóng thì bóng đang ở đâu?")
print("(trong tay cầu thủ? đang bay nhanh? ở xa camera?)")
