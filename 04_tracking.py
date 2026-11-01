"""
BÀI 04 — Theo dõi (tracking): gán cho mỗi người một SỐ ID cố định.

Chạy:   python 04_tracking.py
Kết quả: output/04_tracked_video.mp4  (mỗi người có khung màu riêng + số ID)
         output/04_positions.csv      (frame, id, x, y — vị trí chân từng người)

Khác nhau giữa "phát hiện" và "theo dõi":
- Phát hiện (bài 01–03): mỗi frame AI chỉ nói "ở đây có người".
- Theo dõi (bài này): AI nói thêm "người này vẫn là người số 7 ở frame trước".
  Nhờ vậy ta biết từng cầu thủ chạy đi đâu suốt đoạn video.

Khi hai người chạy cắt nhau hoặc một người bị che, tracker có thể "nhầm" và
đổi số ID. Hãy xem video kết quả và tìm những chỗ như vậy!
"""

import csv  # noqa: I001

import cv2
from ultralytics import YOLO

from config import MODEL_NAME, IMAGE_SIZE, PERSON_CLASS, VIDEO_PATH, OUTPUT_DIR, require_video

# 🔧 THỬ SỬA
CONFIDENCE_THRESHOLD = 0.3
MAX_FRAMES = 300           # None = chạy hết video
SHOW_WINDOW = False        # True = xem trực tiếp (bấm q để dừng)

require_video()

model = YOLO(MODEL_NAME)

video = cv2.VideoCapture(str(VIDEO_PATH))
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_out_path = OUTPUT_DIR / "04_tracked_video.mp4"
writer = cv2.VideoWriter(str(video_out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def color_for_id(track_id):
    """Mỗi ID một màu riêng, và cùng ID thì luôn cùng màu (để mắt dễ theo dõi)."""
    palette = [(255, 99, 71), (50, 205, 50), (30, 144, 255), (255, 215, 0), (238, 130, 238),
               (0, 206, 209), (255, 140, 0), (147, 112, 219), (220, 20, 60), (127, 255, 212)]
    return palette[track_id % len(palette)]


positions = []             # mỗi phần tử: [frame, id, x, y]
frames_per_id = {}         # từ điển: ID → số frame mà ID đó xuất hiện
frame_index = 0

while True:
    ok, frame = video.read()
    if not ok:
        break
    if MAX_FRAMES is not None and frame_index >= MAX_FRAMES:
        break

    # .track() thay cho gọi model(...) trực tiếp.
    # persist=True: "hãy nhớ những người ở frame trước" — thiếu dòng này thì không theo dõi được.
    results = model.track(frame, classes=[PERSON_CLASS], conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE,
                          persist=True, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes

    # Ở frame đầu tiên (hoặc khi không có ai), tracker có thể chưa gán ID → id là None
    if boxes.id is not None:
        for box, track_id in zip(boxes, boxes.id.tolist()):
            track_id = int(track_id)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            # Vị trí CHÂN = điểm giữa cạnh dưới của khung.
            # Ta dùng chân (không dùng tâm khung) vì chân mới chạm mặt sân,
            # sau này sẽ đổi điểm này sang tọa độ thật trên sân (bài homography).
            foot_x = (x1 + x2) // 2
            foot_y = y2

            color = color_for_id(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (foot_x, foot_y), 5, color, -1)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            positions.append([frame_index, track_id, foot_x, foot_y])
            frames_per_id[track_id] = frames_per_id.get(track_id, 0) + 1

    cv2.putText(frame, f"Frame {frame_index}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
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

# Lưu vị trí ra CSV
csv_path = OUTPUT_DIR / "04_positions.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "id", "x", "y"])
    csv_writer.writerows(positions)

# Tóm tắt: ID nào xuất hiện bao lâu?
print(f"\nĐã xử lý {frame_index} frame.")
print(f"Tổng số ID khác nhau: {len(frames_per_id)}")
print("\n15 ID xuất hiện lâu nhất:")
print("  ID | số frame xuất hiện")
print("-----+-------------------")
# sorted(..., key=...) : sắp xếp ID theo số frame, nhiều nhất lên đầu
# [:15] : chỉ lấy 15 phần tử đầu tiên của danh sách
sorted_ids = sorted(frames_per_id, key=frames_per_id.get, reverse=True)
for track_id in sorted_ids[:15]:
    print(f" {track_id:3d} | {frames_per_id[track_id]}")
if len(sorted_ids) > 15:
    print(f" ... và {len(sorted_ids) - 15} ID khác (phần lớn chỉ xuất hiện vài frame)")

print(f"\n✅ Video kết quả: {video_out_path}")
print(f"✅ Vị trí:        {csv_path}")
print("\nCâu hỏi: Trên sân chỉ có khoảng 10–12 người. Nếu tổng số ID lớn hơn nhiều,")
print("điều đó nói lên gì? Hãy tìm trong video 3 lần một người bị ĐỔI ID và ghi lại số frame.")
