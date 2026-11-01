"""
BÀI 01 — Cho AI nhìn MỘT bức ảnh và tìm người.

Chạy:   python 01_detect_image.py
Kết quả: output/01_detected_image.jpg

Bạn sẽ học:
- Mô hình (model) đã huấn luyện sẵn là gì, cách nạp nó bằng 1 dòng code.
- Khung bao (bounding box) = 4 con số x1, y1, x2, y2.
- Độ tin cậy (confidence) = AI "chắc chắn" bao nhiêu, từ 0 đến 1.
"""

import cv2
from ultralytics import YOLO

from config import IMAGE_SIZE, MODEL_NAME, OUTPUT_DIR, PERSON_CLASS, get_test_image

# 🔧 THỬ SỬA: AI chỉ báo những người mà nó tin chắc từ mức này trở lên.
CONFIDENCE_THRESHOLD = 0.3

# ------------------------------------------------------------------
# Bước 1: Nạp mô hình AI (lần đầu sẽ tự tải về, mất vài giây)
# ------------------------------------------------------------------
model = YOLO(MODEL_NAME)

# ------------------------------------------------------------------
# Bước 2: Lấy một bức ảnh
# ------------------------------------------------------------------
# Với máy tính, ảnh chỉ là một bảng số khổng lồ: cao × rộng × 3 màu.
image = get_test_image(frame_index=100)
height, width = image.shape[0], image.shape[1]
print(f"Kích thước ảnh: rộng {width} × cao {height} pixel")

# ------------------------------------------------------------------
# Bước 3: Đưa ảnh cho AI
# ------------------------------------------------------------------
# classes=[PERSON_CLASS]: chỉ tìm người, bỏ qua xe, ghế, chó, mèo...
# conf=CONFIDENCE_THRESHOLD: bỏ những phát hiện mà AI không đủ chắc.
# imgsz=IMAGE_SIZE: kích thước ảnh AI nhìn (xem giải thích trong config.py).
# verbose=False: bảo thư viện đừng in thêm thông tin rườm rà.
results = model(image, classes=[PERSON_CLASS], conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE, verbose=False)

# AI trả về một danh sách, mỗi phần tử ứng với một ảnh.
# Ta chỉ đưa vào 1 ảnh nên lấy phần tử đầu tiên [0].
boxes = results[0].boxes
print(f"\nAI tìm thấy {len(boxes)} người (ngưỡng tin cậy = {CONFIDENCE_THRESHOLD})\n")

# ------------------------------------------------------------------
# Bước 4: Đọc từng khung bao và vẽ lên ảnh
# ------------------------------------------------------------------
print(" STT |   x1    y1    x2    y2 | tin cậy")
print("-----+------------------------+--------")

for number, box in enumerate(boxes, start=1):
    # box.xyxy[0] = [x1, y1, x2, y2]
    #   (x1, y1) = góc TRÊN-TRÁI,  (x2, y2) = góc DƯỚI-PHẢI
    #   Gốc tọa độ (0, 0) nằm ở góc trên-trái của ảnh; y càng lớn càng đi XUỐNG.
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)    # pixel phải là số nguyên
    confidence = float(box.conf[0])

    print(f" {number:3d} | {x1:5d} {y1:5d} {x2:5d} {y2:5d} |  {confidence:.2f}")

    # Màu trong OpenCV viết theo thứ tự (Xanh dương, Xanh lá, Đỏ) — gọi là BGR
    green = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), green, 2)

    # Ghi số thứ tự và độ tin cậy phía trên khung.
    # Lưu ý: cv2.putText không viết được chữ có dấu tiếng Việt, nên chỉ ghi số.
    label = f"#{number} {confidence:.2f}"
    cv2.putText(image, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, green, 2)

# ------------------------------------------------------------------
# Bước 5: Lưu ảnh kết quả
# ------------------------------------------------------------------
output_path = OUTPUT_DIR / "01_detected_image.jpg"
cv2.imwrite(str(output_path), image)
print(f"\n✅ Đã lưu ảnh: {output_path}")
print("   Mở ảnh ra xem: AI có bỏ sót ai không? Có nhầm vật gì thành người không?")
