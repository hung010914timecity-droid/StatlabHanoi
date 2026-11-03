"""
BÀI 04 — Lấy quỹ đạo của RIÊNG cầu thủ mục tiêu, vẽ video và đường chạy.

Chạy:   python 04_player_track.py     (cần chạy bài 01 và 03 trước)
Kết quả: output/player_track.csv       (frame, t, id, x1, y1, x2, y2, foot_x, foot_y)
         output/04_target_video.mp4    (chỉ cầu thủ mục tiêu có khung đỏ + vệt chân 2 giây gần nhất)
         output/04_trail.png           (toàn bộ đường chạy vẽ trên một frame)

Bài này không chạy AI: chỉ ghép file all_tracks.csv (bài 01) với target.json (bài 03).

player_track.csv chính là "dữ liệu vị trí" của SPEC (FR-14). Hiện tại vị trí đang tính
bằng PIXEL trên ảnh camera. Buổi 7 (homography) sẽ đổi pixel thành MÉT trên sân thật,
khi đó mới tính được quãng đường, tốc độ, heatmap.
"""

import csv

import cv2

from config import META_JSON, OUTPUT_DIR, PLAYER_CSV, TARGET_JSON, TRACKS_CSV, VIDEO_PATH, require_file, require_video
from tools import build_target, foot_point, group_by_frame, load_json, read_tracks

# 🔧 THỬ SỬA
TRAIL_SECONDS = 2          # vệt chân trong video dài bao nhiêu giây
SHOW_OTHERS = True         # True = vẽ mờ khung của những người khác để so sánh

require_video()
require_file(TRACKS_CSV, "01_track_video.py")
require_file(TARGET_JSON, "03_pick_player.py")

meta = load_json(META_JSON)
fps = meta["fps"]
n_frames = meta["frames_processed"]
segments = load_json(TARGET_JSON)["segments"]
by_frame = group_by_frame(read_tracks(TRACKS_CSV))

# ------------------------------------------------------------------
# Bước 1: Lọc ra các khung của cầu thủ mục tiêu, lưu thành player_track.csv
# ------------------------------------------------------------------
target_rows = build_target(by_frame, segments, n_frames)
if not target_rows:
    print("❌ Không tìm thấy khung nào của cầu thủ đã chọn. Hãy chạy lại 03_pick_player.py.")
    raise SystemExit(1)

with open(PLAYER_CSV, "w", newline="", encoding="utf-8") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "t", "id", "x1", "y1", "x2", "y2", "foot_x", "foot_y"])
    for row in target_rows:
        foot_x, foot_y = foot_point(row)
        csv_writer.writerow([row["frame"], round(row["frame"] / fps, 3), row["id"],
                             row["x1"], row["y1"], row["x2"], row["y2"], round(foot_x, 1), round(foot_y, 1)])

# ------------------------------------------------------------------
# Bước 2: Mục tiêu bị mất dấu ở những đoạn nào?
# ------------------------------------------------------------------
# Một "đoạn mất dấu" = nhiều frame liên tiếp không thấy mục tiêu.
visible_frames = {row["frame"] for row in target_rows}
gaps = []                  # mỗi phần tử: (frame bắt đầu mất, frame cuối cùng còn mất)
gap_start = None
for frame in range(n_frames):
    if frame not in visible_frames and gap_start is None:
        gap_start = frame                       # bắt đầu mất
    elif frame in visible_frames and gap_start is not None:
        gaps.append((gap_start, frame - 1))     # vừa tìm lại được
        gap_start = None
if gap_start is not None:
    gaps.append((gap_start, n_frames - 1))      # mất tới tận cuối video

coverage = 100 * len(visible_frames) / n_frames
print(f"Cầu thủ mục tiêu có mặt ở {len(visible_frames)}/{n_frames} frame = {coverage:.0f}%")
print(f"Các ID đã dùng cho cầu thủ này: {sorted({row['id'] for row in target_rows})}")
if gaps:
    print(f"\nCó {len(gaps)} đoạn mất dấu:")
    for start, end in gaps[:15]:
        print(f"  frame {start:4d}–{end:4d}  ({start / fps:5.1f}s → {end / fps:5.1f}s, {end - start + 1} frame)")
    if len(gaps) > 15:
        print(f"  ... và {len(gaps) - 15} đoạn khác")
else:
    print("Không mất dấu lần nào. Tuyệt!")

# ------------------------------------------------------------------
# Bước 3: Vẽ video chỉ đánh dấu cầu thủ mục tiêu
# ------------------------------------------------------------------
target_by_frame = {row["frame"]: row for row in target_rows}
trail_length = int(TRAIL_SECONDS * fps)

video = cv2.VideoCapture(str(VIDEO_PATH))
video_out_path = OUTPUT_DIR / "04_target_video.mp4"
writer = cv2.VideoWriter(str(video_out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (meta["width"], meta["height"]))

trail = []                 # vị trí chân gần đây: danh sách (frame, x, y)
first_image = None

for frame in range(n_frames):
    ok, image = video.read()
    if not ok:
        break
    if first_image is None:
        first_image = image.copy()

    if SHOW_OTHERS:
        for row in by_frame.get(frame, []):
            cv2.rectangle(image, (int(row["x1"]), int(row["y1"])), (int(row["x2"]), int(row["y2"])), (160, 160, 160), 1)

    target = target_by_frame.get(frame)
    if target is not None:
        x1, y1, x2, y2 = int(target["x1"]), int(target["y1"]), int(target["x2"]), int(target["y2"])
        foot_x, foot_y = foot_point(target)
        trail.append((frame, int(foot_x), int(foot_y)))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(image, f"MUC TIEU (ID {target['id']})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    else:
        cv2.putText(image, "MAT DAU", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Chỉ giữ vệt của TRAIL_SECONDS giây gần nhất
    trail = [point for point in trail if frame - point[0] <= trail_length]
    for (_, xa, ya), (_, xb, yb) in zip(trail, trail[1:]):     # nối từng cặp điểm liên tiếp
        cv2.line(image, (xa, ya), (xb, yb), (0, 255, 255), 3)

    cv2.putText(image, f"Frame {frame}  {frame / fps:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    writer.write(image)

video.release()
writer.release()

# ------------------------------------------------------------------
# Bước 4: Vẽ TOÀN BỘ đường chạy lên frame đầu tiên
# ------------------------------------------------------------------
trail_image = first_image
points = [(row["frame"], foot_point(row)) for row in target_rows]
for (frame_a, (xa, ya)), (frame_b, (xb, yb)) in zip(points, points[1:]):
    # Nếu giữa hai điểm có đoạn mất dấu dài (> 0.5 giây) thì không nối, để khỏi vẽ một nét "nhảy cóc"
    if frame_b - frame_a <= fps / 2:
        cv2.line(trail_image, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 255), 3)

start_point = (int(points[0][1][0]), int(points[0][1][1]))
end_point = (int(points[-1][1][0]), int(points[-1][1][1]))
cv2.circle(trail_image, start_point, 12, (0, 255, 0), -1)     # chấm xanh lá = điểm bắt đầu
cv2.circle(trail_image, end_point, 12, (0, 0, 255), -1)       # chấm đỏ = điểm kết thúc
cv2.putText(trail_image, "BAT DAU", (start_point[0] + 15, start_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.putText(trail_image, "KET THUC", (end_point[0] + 15, end_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

trail_path = OUTPUT_DIR / "04_trail.png"
cv2.imwrite(str(trail_path), trail_image)

print(f"\n✅ Quỹ đạo (pixel): {PLAYER_CSV}")
print(f"✅ Video mục tiêu:  {video_out_path}")
print(f"✅ Đường chạy:      {trail_path}")
print("\nCâu hỏi:")
print("  1. Xem video: khung đỏ có lúc nào nhảy sang NGƯỜI KHÁC không? Nếu có, chạy lại bài 03 để sửa.")
print("  2. Trong ảnh đường chạy, cầu thủ ở gần camera đi 1 mét thì vạch dài hơn hay ngắn hơn ở xa camera?")
print("     → Vì vậy chưa thể dùng pixel để tính quãng đường. Cần homography (buổi 7).")
