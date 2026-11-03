"""
BÀI 03 — Click chuột để chọn CẦU THỦ MỤC TIÊU (và chọn lại khi tracker đổi ID).

Chạy:   python 03_pick_player.py      (cần chạy bài 01 trước)
Kết quả: output/target.json            (bạn đã chọn ai, từ frame nào)

Một cửa sổ sẽ mở ra, hiện một frame của video với khung + ID của mọi người.
  Click chuột trái vào một người  → người đó là MỤC TIÊU (khung đỏ) từ frame này trở đi
  d / a                           → tới / lùi 1 frame
  f / s                           → tới / lùi 1 giây
  n                               → nhảy tới chỗ tiếp theo mà mục tiêu bị MẤT DẤU
  u                               → hoàn tác lần click gần nhất
  q (hoặc Esc)                    → lưu và thoát
(Bấm phím khi cửa sổ video đang được chọn, không phải cửa sổ Terminal.)

Vì sao cần chọn lại? Tracker hay đổi ID khi cầu thủ bị che (xem bài 02).
Thay vì viết thuật toán phức tạp, ta để NGƯỜI sửa: thấy mục tiêu mất dấu
thì bấm n để tới đó, tua tới lúc cầu thủ hiện lại, click vào họ lần nữa.
Hệ thống thật (SPEC FR-13) cũng cho người dùng "click chọn lại ở bất kỳ frame".

Mỗi lần click được lưu thành một cặp [frame, ID] nghĩa là
"từ frame này trở đi, cầu thủ của tôi mang số ID này".
"""

import cv2

from config import META_JSON, TARGET_JSON, TRACKS_CSV, VIDEO_PATH, require_file, require_video
from tools import find_player_at, group_by_frame, load_json, read_tracks, save_json, target_id_at

# 🔧 THỬ SỬA: cửa sổ rộng tối đa bao nhiêu pixel (giảm nếu màn hình nhỏ)
DISPLAY_WIDTH = 1280

require_video()
require_file(TRACKS_CSV, "01_track_video.py")

meta = load_json(META_JSON)
fps = meta["fps"]
last_frame = meta["frames_processed"] - 1
by_frame = group_by_frame(read_tracks(TRACKS_CSV))

# Nếu đã chọn từ lần chạy trước thì nạp lại để sửa tiếp
segments = []              # danh sách [frame, ID], theo thứ tự bạn click
if TARGET_JSON.exists():
    segments = load_json(TARGET_JSON)["segments"]
    print(f"Đã nạp {len(segments)} lần chọn từ lần trước: {segments}  (bấm u để xóa dần)")

# Bắt đầu ở frame có nhiều người nhất → dễ nhìn thấy cầu thủ mình muốn chọn
current_frame = max(by_frame, key=lambda frame: len(by_frame[frame]))

video = cv2.VideoCapture(str(VIDEO_PATH))
scale = min(1.0, DISPLAY_WIDTH / meta["width"])     # thu nhỏ để vừa màn hình
WINDOW = "StatsLab - Chon cau thu (q = luu & thoat)"


# ------------------------------------------------------------------
# Các hàm nhỏ
# ------------------------------------------------------------------
def read_frame(frame_index):
    """Nhảy tới frame cần xem trong video rồi đọc nó."""
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, image = video.read()
    if not ok:
        raise RuntimeError(f"Không đọc được frame {frame_index}")
    return image


def target_visible(frame_index):
    """Ở frame này, tracker có thấy cầu thủ mục tiêu không?"""
    if not segments:
        return False
    wanted_id = target_id_at(segments, frame_index)
    return any(row["id"] == wanted_id for row in by_frame.get(frame_index, []))


def next_lost_frame(after_frame):
    """Tìm frame tiếp theo mà mục tiêu VỪA biến mất (frame trước còn thấy, frame này không)."""
    for frame_index in range(after_frame + 1, last_frame + 1):
        if target_visible(frame_index - 1) and not target_visible(frame_index):
            return frame_index
    return None


def draw(frame_index):
    """Vẽ khung mọi người (mục tiêu màu đỏ), thanh thông tin trên và hướng dẫn phím dưới."""
    image = read_frame(frame_index)
    wanted_id = target_id_at(segments, frame_index) if segments else None

    for row in by_frame.get(frame_index, []):
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        if row["id"] == wanted_id:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
            cv2.putText(image, f"MUC TIEU #{row['id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"#{row['id']}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Thu nhỏ SAU khi vẽ khung, rồi mới ghi chữ để chữ không bị nhòe
    image = cv2.resize(image, None, fx=scale, fy=scale)
    width = image.shape[1]

    if wanted_id is None:
        status, status_color = "Chua chon ai - hay click vao mot nguoi", (0, 255, 255)
    elif target_visible(frame_index):
        status, status_color = f"Muc tieu: ID {wanted_id} - dang theo doi", (0, 255, 0)
    else:
        status, status_color = f"Muc tieu: ID {wanted_id} - MAT DAU! Tua toi luc hien lai roi click", (0, 0, 255)

    cv2.rectangle(image, (0, 0), (width, 40), (0, 0, 0), -1)
    cv2.putText(image, f"Frame {frame_index}/{last_frame}  {frame_index / fps:.1f}s   {status}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.rectangle(image, (0, image.shape[0] - 32), (width, image.shape[0]), (0, 0, 0), -1)
    cv2.putText(image, "Click: chon | a/d: -/+1 frame | s/f: -/+1 giay | n: cho mat dau tiep | u: hoan tac | q: luu",
                (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return image


# Khi bạn click, OpenCV gọi hàm này. Ta chỉ ghi lại điểm click (đổi về kích thước gốc),
# vòng lặp chính bên dưới sẽ xử lý.
clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x / scale, y / scale))


# ------------------------------------------------------------------
# Vòng lặp chính: vẽ → chờ phím/click → cập nhật
# ------------------------------------------------------------------
cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(WINDOW, on_mouse)
print("\nCửa sổ đã mở. Click vào cầu thủ bạn muốn theo dõi. Bấm q để lưu và thoát.")

need_redraw = True
while True:
    if need_redraw:
        cv2.imshow(WINDOW, draw(current_frame))
        need_redraw = False

    key = cv2.waitKey(30) & 0xFF       # chờ 30 ms xem có phím nào được bấm không

    # Người dùng bấm nút X đóng cửa sổ → coi như bấm q
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break

    if clicks:
        x, y = clicks.pop()
        clicks.clear()
        picked_id = find_player_at(by_frame.get(current_frame, []), x, y)
        if picked_id is None:
            print(f"  Frame {current_frame}: chỗ bạn click không có ai. Click vào bên trong một khung.")
        else:
            # Ở frame này đã chọn rồi thì thay bằng lựa chọn mới
            segments = [seg for seg in segments if seg[0] != current_frame]
            segments.append([current_frame, picked_id])
            print(f"  Từ frame {current_frame} ({current_frame / fps:.1f}s): mục tiêu = ID {picked_id}")
        need_redraw = True

    if key in (ord("q"), 27):          # 27 = phím Esc
        break
    elif key == ord("d"):
        current_frame = min(current_frame + 1, last_frame)
        need_redraw = True
    elif key == ord("a"):
        current_frame = max(current_frame - 1, 0)
        need_redraw = True
    elif key == ord("f"):
        current_frame = min(current_frame + round(fps), last_frame)
        need_redraw = True
    elif key == ord("s"):
        current_frame = max(current_frame - round(fps), 0)
        need_redraw = True
    elif key == ord("n"):
        lost = next_lost_frame(current_frame)
        if lost is None:
            print("  Từ đây tới cuối video, mục tiêu không bị mất dấu lần nào nữa. 🎉")
        else:
            current_frame = lost
            print(f"  Mục tiêu mất dấu ở frame {lost} ({lost / fps:.1f}s). Bấm d/f tua tới lúc họ hiện lại rồi click.")
        need_redraw = True
    elif key == ord("u") and segments:
        removed = segments.pop()
        print(f"  Đã hoàn tác: [frame {removed[0]}, ID {removed[1]}]")
        need_redraw = True

video.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------------
# Lưu kết quả
# ------------------------------------------------------------------
if not segments:
    print("\nBạn chưa chọn ai nên chưa lưu gì. Chạy lại và click vào một cầu thủ nhé.")
    raise SystemExit(0)

segments = sorted(segments)
target = {"video": meta["video"], "fps": fps, "n_frames": meta["n_frames"], "segments": segments}
save_json(target, TARGET_JSON)

print("\nCác lần chọn (frame bắt đầu → ID):")
for start, track_id in segments:
    print(f"  frame {start:5d} ({start / fps:5.1f}s) → ID {track_id}")
print(f"\n✅ Đã lưu: {TARGET_JSON}")
print("Tiếp theo: python 04_player_track.py  để lấy quỹ đạo và video của riêng cầu thủ này.")
