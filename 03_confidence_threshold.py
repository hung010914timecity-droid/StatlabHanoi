"""
BÀI 03 — Thử nghiệm: đổi NGƯỠNG TIN CẬY thì AI thấy bao nhiêu người?

Chạy:   python 03_confidence_threshold.py
Kết quả: output/03_threshold_compare.jpg  (4 ảnh ghép lại, mỗi ảnh một ngưỡng)

Ý tưởng:
- Ngưỡng THẤP  → AI báo nhiều khung hơn: ít bỏ sót người, nhưng dễ NHẦM
  (ví dụ nhầm cái ghế, cái bóng thành người).
- Ngưỡng CAO   → AI chỉ báo khi rất chắc: ít nhầm, nhưng dễ BỎ SÓT
  (người ở xa, bị che một nửa).
Không có ngưỡng nào hoàn hảo — ta phải chọn mức cân bằng cho bài toán của mình.
"""

import cv2
import numpy as np
from ultralytics import YOLO

from config import IMAGE_SIZE, MODEL_NAME, OUTPUT_DIR, PERSON_CLASS, get_test_image

# 🔧 THỬ SỬA: 4 ngưỡng muốn so sánh (giữ đúng 4 số để ghép thành lưới 2×2)
THRESHOLDS = [0.1, 0.3, 0.5, 0.7]

model = YOLO(MODEL_NAME)
original = get_test_image(frame_index=100)

# Chạy AI MỘT lần với ngưỡng thấp nhất để lấy hết các khung,
# sau đó tự lọc theo từng ngưỡng. Nhanh hơn chạy AI 4 lần.
results = model(original, classes=[PERSON_CLASS], conf=min(THRESHOLDS), imgsz=IMAGE_SIZE, verbose=False)
boxes = results[0].boxes

print("\nNgưỡng | Số người AI thấy")
print("-------+-----------------")

panels = []
for threshold in THRESHOLDS:
    image = original.copy()       # copy để không vẽ chồng lên ảnh gốc
    count = 0
    for box in boxes:
        confidence = float(box.conf[0])
        if confidence < threshold:    # không đủ chắc → bỏ qua
            continue
        count += 1
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    print(f"  {threshold:.1f}  | {count}")

    # Ghi tiêu đề lên ảnh: nền đen, chữ vàng cho dễ đọc
    cv2.rectangle(image, (0, 0), (image.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(image, f"conf >= {threshold}: {count} nguoi", (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    panels.append(image)

# Ghép 4 ảnh thành lưới 2×2:  [ảnh 1 | ảnh 2]
#                              [ảnh 3 | ảnh 4]
top_row = np.hstack([panels[0], panels[1]])       # hstack = ghép ngang
bottom_row = np.hstack([panels[2], panels[3]])
grid = np.vstack([top_row, bottom_row])           # vstack = ghép dọc

# Thu nhỏ một nửa cho file không quá nặng
grid = cv2.resize(grid, None, fx=0.5, fy=0.5)

output_path = OUTPUT_DIR / "03_threshold_compare.jpg"
cv2.imwrite(str(output_path), grid)
print(f"\n✅ Đã lưu: {output_path}")
print("\nCâu hỏi: Bạn tự đếm có bao nhiêu người trong ảnh? Ngưỡng nào cho con số gần đúng nhất?")
