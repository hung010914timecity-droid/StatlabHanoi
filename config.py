"""
config.py — Các cài đặt dùng chung cho mọi bài trong thư mục yolo/.

File này KHÔNG chạy trực tiếp. Các bài 01, 02, 03... sẽ "nhập" (import) nó
để dùng chung tên video, tên mô hình và thư mục lưu kết quả.
Muốn đổi video hay đổi mô hình? Chỉ cần sửa ở đây một lần.
"""

from pathlib import Path
import urllib.request

import cv2

# ------------------------------------------------------------------
# 1. Thư mục
# ------------------------------------------------------------------
# Path(__file__).parent = thư mục đang chứa file này (tức là yolo/).
# Nhờ vậy chạy script từ đâu cũng tìm đúng file.
BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "output"

# Tạo thư mục nếu chưa có (exist_ok=True: có rồi thì thôi, không báo lỗi)
VIDEO_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 2. Video đầu vào
# ------------------------------------------------------------------
# Đặt clip bóng rổ 30–60 giây vào thư mục yolo/videos/ và đặt tên clip1.mp4
# (hoặc sửa tên ở dòng dưới cho khớp với file của bạn).
VIDEO_PATH = VIDEO_DIR / "clip1.mp4"

# ------------------------------------------------------------------
# 3. Mô hình AI
# ------------------------------------------------------------------
# YOLO26 là thế hệ YOLO mới nhất của Ultralytics.
# "n" = nano: bản nhỏ nhất, nhanh nhất, chạy tốt trên máy không có GPU.
# Lần chạy đầu tiên, thư viện sẽ tự tải file mô hình (~6 MB) từ Internet.
# Muốn chính xác hơn (nhưng chậm hơn) thử: "yolo26s.pt" hoặc "yolo26m.pt".
MODEL_NAME = "yolo26n.pt"

# Trước khi "nhìn", YOLO thu nhỏ ảnh xuống cạnh dài = số này (pixel).
# Video 1920 pixel mà thu xuống 640 thì cầu thủ ở xa chỉ còn vài pixel → AI bỏ sót.
# Thử trên video VBA mẫu (frame 100):  640 → 6 người,  1280 → 21 người,  1920 → 32 người.
# Số càng lớn càng thấy rõ người nhỏ, nhưng chạy càng chậm (1280 chậm hơn 640 khoảng 3 lần).
IMAGE_SIZE = 1280

# Mỗi loại vật thể mà YOLO biết có một con số (gọi là "class").
# Mô hình có sẵn được học trên bộ ảnh COCO gồm 80 loại vật thể.
PERSON_CLASS = 0   # 0 = person (người)
BALL_CLASS = 32    # 32 = sports ball (quả bóng thể thao)

# Ảnh mẫu dùng khi bạn chưa có video (ảnh xe buýt có vài người đứng trước)
SAMPLE_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_IMAGE_PATH = VIDEO_DIR / "sample.jpg"


def get_test_image(frame_index=100):
    """
    Trả về MỘT bức ảnh để thử AI.
    - Nếu đã có video: lấy frame số `frame_index` trong video.
    - Nếu chưa có video: tải ảnh mẫu từ Internet về và dùng ảnh đó.
    """
    if VIDEO_PATH.exists():
        video = cv2.VideoCapture(str(VIDEO_PATH))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # Nếu video ngắn hơn số frame muốn lấy thì lấy frame ở giữa video
        if frame_index >= total_frames:
            frame_index = total_frames // 2
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)   # nhảy tới frame cần lấy
        ok, image = video.read()
        video.release()
        if not ok:
            raise RuntimeError(f"Không đọc được frame {frame_index} trong {VIDEO_PATH}")
        print(f"Dùng frame {frame_index} của video {VIDEO_PATH.name}")
        return image

    if not SAMPLE_IMAGE_PATH.exists():
        print("Chưa có video → tải ảnh mẫu về để thử...")
        urllib.request.urlretrieve(SAMPLE_IMAGE_URL, SAMPLE_IMAGE_PATH)
    print(f"Dùng ảnh mẫu {SAMPLE_IMAGE_PATH.name} (chưa thấy file {VIDEO_PATH.name} trong thư mục videos/)")
    return cv2.imread(str(SAMPLE_IMAGE_PATH))


def require_video():
    """Dừng chương trình kèm lời nhắc nếu chưa có video."""
    if not VIDEO_PATH.exists():
        print(f"❌ Chưa có video: {VIDEO_PATH}")
        print("   Hãy chép clip bóng rổ vào thư mục yolo/videos/ và đặt tên clip1.mp4")
        print("   (hoặc sửa biến VIDEO_PATH trong file config.py).")
        raise SystemExit(1)
