"""
tools.py — Các hàm dùng chung cho bài 02–05 (đọc file kết quả, chọn cầu thủ, tô màu).

File này KHÔNG chạy trực tiếp. Mỗi hàm đều ngắn, bạn có thể mở ra đọc
khi muốn biết một bài "bên trong" làm gì.
"""

import csv
import json


def read_tracks(csv_path):
    """
    Đọc all_tracks.csv, trả về một danh sách. Mỗi phần tử là một từ điển (dict), ví dụ:
        {"frame": 12, "id": 7, "x1": 640.0, "y1": 300.0, "x2": 700.0, "y2": 460.0, "conf": 0.81}
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # csv đọc mọi thứ thành chữ (str) → phải đổi lại thành số
            rows.append({
                "frame": int(row["frame"]),
                "id": int(row["id"]),
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
                "conf": float(row["conf"]),
            })
    return rows


def group_by_frame(tracks):
    """Gom các dòng theo frame: {số frame: [các khung trong frame đó]}. Tra cứu nhanh hơn nhiều."""
    by_frame = {}
    for row in tracks:
        by_frame.setdefault(row["frame"], []).append(row)
    return by_frame


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def color_for_id(track_id):
    """Mỗi ID một màu riêng, và cùng ID thì luôn cùng màu (giống yolo/04_tracking.py)."""
    palette = [(255, 99, 71), (50, 205, 50), (30, 144, 255), (255, 215, 0), (238, 130, 238),
               (0, 206, 209), (255, 140, 0), (147, 112, 219), (220, 20, 60), (127, 255, 212)]
    return palette[track_id % len(palette)]


def foot_point(row):
    """Vị trí CHÂN = điểm giữa cạnh dưới của khung (chân mới là chỗ chạm mặt sân)."""
    return (row["x1"] + row["x2"]) / 2, row["y2"]


def find_player_at(boxes, x, y):
    """
    Trả về ID của người có khung chứa điểm (x, y). Không có ai ở đó thì trả về None.
    Nếu điểm nằm trong nhiều khung chồng nhau, chọn khung NHỎ nhất
    (thường là người đứng xa camera, bị người gần camera che một phần).
    """
    best_id = None
    best_area = None
    for row in boxes:
        inside = row["x1"] <= x <= row["x2"] and row["y1"] <= y <= row["y2"]
        if not inside:
            continue
        area = (row["x2"] - row["x1"]) * (row["y2"] - row["y1"])
        if best_area is None or area < best_area:
            best_id, best_area = row["id"], area
    return best_id


def target_id_at(segments, frame):
    """
    segments = [[frame_bắt_đầu, id], ...] — các lần bạn click chọn cầu thủ.
    "Từ frame_bắt_đầu trở đi, cầu thủ mục tiêu mang số id" (cho tới lần chọn tiếp theo).
    Hàm trả về ID của cầu thủ mục tiêu ở `frame`.
    Lần chọn đầu tiên cũng áp dụng cho các frame trước nó.
    """
    segments = sorted(segments)
    current_id = segments[0][1]
    for start, track_id in segments:
        if frame >= start:
            current_id = track_id
    return current_id


def build_target(by_frame, segments, n_frames):
    """Lấy ra các khung của RIÊNG cầu thủ mục tiêu, frame nào có thì lấy (mỗi frame tối đa 1 khung)."""
    target_rows = []
    for frame in range(n_frames):
        wanted_id = target_id_at(segments, frame)
        for row in by_frame.get(frame, []):
            if row["id"] == wanted_id:
                target_rows.append(row)
                break
    return target_rows
