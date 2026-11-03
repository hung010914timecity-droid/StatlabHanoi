"""
BÀI 05 (thử thách) — Tự viết một TRACKER đơn giản bằng IoU.

Chạy:   python 05_my_tracker.py       (cần chạy bài 01 trước)
Kết quả: output/05_my_tracks.csv       (cùng dạng all_tracks.csv, nhưng ID do tracker CỦA BẠN gán)
         output/05_my_tracker.mp4      (video với ID của tracker của bạn)

Mục đích: hiểu tracker "bên trong" làm gì. Không cần AI: ta lấy lại các khung
đã phát hiện ở bài 01, XÓA hết ID của ByteTrack, rồi tự gán ID lại từ đầu.

IoU (Intersection over Union — "phần chung chia phần gộp") đo hai khung chồng nhau bao nhiêu:
    IoU = diện tích phần CHUNG / diện tích phần GỘP của hai khung
    = 1   : trùng khít
    = 0   : không chạm nhau
Giữa hai frame liên tiếp (1/25 giây) một người chỉ dịch đi rất ít, nên khung của họ
ở frame sau chồng lên khung ở frame trước rất nhiều → IoU cao → "chắc là cùng một người".

Thuật toán (mỗi frame):
  1. Tính IoU giữa MỌI người đang theo dõi (khung cuối cùng của họ) và MỌI khung mới.
  2. Ghép các cặp có IoU cao nhất trước (tham lam). Mỗi người, mỗi khung chỉ ghép một lần.
     Cặp có IoU < IOU_MIN thì không ghép.
  3. Khung mới chưa ghép với ai → cấp ID mới.
  4. Người không được ghép → tăng bộ đếm "mất tích". Mất tích quá MAX_MISSING frame thì quên luôn.
ByteTrack làm giống hệt các bước trên, cộng thêm: đoán trước vị trí người sẽ tới (bộ lọc Kalman)
và cho khung có độ tin cậy thấp một cơ hội ghép lần hai.
"""

import csv

import cv2

from config import META_JSON, OUTPUT_DIR, TRACKS_CSV, VIDEO_PATH, require_file, require_video
from tools import color_for_id, group_by_frame, load_json, read_tracks

# 🔧 THỬ SỬA
IOU_MIN = 0.3              # IoU tối thiểu để coi là cùng một người (thử 0.1 và 0.6)
MAX_MISSING = 10           # số frame được phép "mất tích" trước khi bị quên (thử 0 và 50)
MIN_SECONDS = 3            # ID sống ít nhất bao lâu thì tính là "bền" (giống bài 02)
MAKE_VIDEO = True          # False = bỏ qua bước vẽ video cho nhanh


def iou(box_a, box_b):
    """IoU của hai khung, mỗi khung là (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    # Phần chung cũng là một hình chữ nhật. Nếu hai khung không chạm nhau thì rộng/cao = 0.
    overlap_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0, min(ay2, by2) - max(ay1, by1))
    overlap = overlap_w * overlap_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
    return overlap / union if union > 0 else 0.0


# Kiểm tra nhanh hàm iou trước khi dùng
assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0          # trùng khít
assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0        # xa nhau
assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9   # chồng một nửa: chung 50, gộp 150

require_video()
require_file(TRACKS_CSV, "01_track_video.py")

meta = load_json(META_JSON)
fps = meta["fps"]
n_frames = meta["frames_processed"]
bytetrack_rows = read_tracks(TRACKS_CSV)
by_frame = group_by_frame(bytetrack_rows)

# ------------------------------------------------------------------
# Tracker của bạn
# ------------------------------------------------------------------
alive = {}                 # những người đang theo dõi: ID → {"box": khung cuối, "missing": số frame mất tích}
next_id = 1
my_rows = []               # kết quả: [frame, id, x1, y1, x2, y2, conf]

for frame in range(n_frames):
    # Lấy các khung của frame này, BỎ ID của ByteTrack đi
    detections = by_frame.get(frame, [])
    boxes = [(row["x1"], row["y1"], row["x2"], row["y2"]) for row in detections]

    # Bước 1: IoU của mọi cặp (người đang theo dõi, khung mới)
    pairs = []
    for track_id, info in alive.items():
        for box_index, box in enumerate(boxes):
            score = iou(info["box"], box)
            if score >= IOU_MIN:
                pairs.append((score, track_id, box_index))

    # Bước 2: ghép cặp IoU cao nhất trước
    pairs.sort(reverse=True)               # sắp xếp giảm dần theo score (phần tử đầu của mỗi cặp)
    matched_ids = set()
    matched_boxes = {}                     # chỉ số khung → ID được gán
    for score, track_id, box_index in pairs:
        if track_id in matched_ids or box_index in matched_boxes:
            continue                       # người này hoặc khung này đã được ghép rồi
        matched_ids.add(track_id)
        matched_boxes[box_index] = track_id

    # Bước 3: khung chưa ghép → ID mới
    for box_index in range(len(boxes)):
        if box_index not in matched_boxes:
            matched_boxes[box_index] = next_id
            next_id += 1

    # Cập nhật "khung cuối" của mọi người vừa được thấy, và lưu kết quả
    for box_index, track_id in matched_boxes.items():
        alive[track_id] = {"box": boxes[box_index], "missing": 0}
        row = detections[box_index]
        my_rows.append([frame, track_id, row["x1"], row["y1"], row["x2"], row["y2"], row["conf"]])

    # Bước 4: người không thấy ở frame này → tăng mất tích, quá lâu thì quên
    for track_id in list(alive):           # list(...) vì ta xóa phần tử trong lúc duyệt
        if track_id not in matched_boxes.values():
            alive[track_id]["missing"] += 1
            if alive[track_id]["missing"] > MAX_MISSING:
                del alive[track_id]

# ------------------------------------------------------------------
# So sánh với ByteTrack
# ------------------------------------------------------------------
def count_ids(id_list):
    """id_list = ID của từng khung. Trả về (tổng số ID, số ID bền)."""
    frames_per_id = {}
    for track_id in id_list:
        frames_per_id[track_id] = frames_per_id.get(track_id, 0) + 1
    durable = sum(1 for count in frames_per_id.values() if count / fps >= MIN_SECONDS)
    return len(frames_per_id), durable


bytetrack_total, bytetrack_durable = count_ids([row["id"] for row in bytetrack_rows])
my_total, my_durable = count_ids([row[1] for row in my_rows])

print(f"Cài đặt: IOU_MIN = {IOU_MIN}, MAX_MISSING = {MAX_MISSING}")
print(f"\n                 | Tổng số ID | ID bền (≥ {MIN_SECONDS}s)")
print("-----------------+------------+--------------")
print(f" ByteTrack       | {bytetrack_total:10d} | {bytetrack_durable:12d}")
print(f" Tracker của bạn | {my_total:10d} | {my_durable:12d}")

my_csv = OUTPUT_DIR / "05_my_tracks.csv"
with open(my_csv, "w", newline="", encoding="utf-8") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "id", "x1", "y1", "x2", "y2", "conf"])
    csv_writer.writerows(my_rows)
print(f"\n✅ ID của bạn: {my_csv}")

# ------------------------------------------------------------------
# Vẽ video với ID của tracker của bạn
# ------------------------------------------------------------------
if MAKE_VIDEO:
    my_by_frame = {}
    for row in my_rows:
        my_by_frame.setdefault(row[0], []).append(row)

    video = cv2.VideoCapture(str(VIDEO_PATH))
    video_out_path = OUTPUT_DIR / "05_my_tracker.mp4"
    writer = cv2.VideoWriter(str(video_out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (meta["width"], meta["height"]))
    for frame in range(n_frames):
        ok, image = video.read()
        if not ok:
            break
        for _, track_id, x1, y1, x2, y2, _ in my_by_frame.get(frame, []):
            color = color_for_id(track_id)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, f"ID {track_id}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Tracker cua ban - Frame {frame}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        writer.write(image)
    video.release()
    writer.release()
    print(f"✅ Video:      {video_out_path}")

print("\nCâu hỏi:")
print("  1. Tracker của bạn có nhiều ID hơn hay ít hơn ByteTrack? Vì sao?")
print("  2. Đặt MAX_MISSING = 0. Số ID tăng hay giảm? Giải thích bằng lời.")
print("  3. Khi hai cầu thủ chạy cắt nhau, IoU giúp được gì và không giúp được gì?")
print("     (Gợi ý: nhìn khung của hai người ngay lúc họ chồng lên nhau.)")
