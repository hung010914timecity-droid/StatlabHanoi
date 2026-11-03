"""
BÀI 02 — Tracker làm tốt đến đâu? Kiểm tra "tuổi thọ" của từng ID.

Chạy:   python 02_id_report.py        (cần chạy bài 01 trước)
Kết quả: in bảng thống kê ra màn hình
         output/02_id_timeline.png    (biểu đồ: mỗi ID là một vạch ngang, từ lúc xuất hiện tới lúc biến mất)

Bài này KHÔNG chạy AI, chỉ đọc file all_tracks.csv nên chạy rất nhanh.

Một tracker hoàn hảo trên sân có 10 cầu thủ sẽ cho đúng ~10 ID (+ trọng tài),
mỗi ID sống từ đầu đến cuối video. Thực tế thì không:
- Người bị che → tracker mất dấu → khi hiện lại, người đó nhận ID MỚI.
  Một người thật bị "cắt" thành nhiều mảnh ID ngắn.
- Ta đo mức "vỡ vụn" bằng: độ phân mảnh = số ID / số người thật.
  Bằng 1 là hoàn hảo; càng lớn càng tệ.
"""

import matplotlib.pyplot as plt

from config import META_JSON, OUTPUT_DIR, TRACKS_CSV, require_file
from tools import group_by_frame, load_json, read_tracks

# 🔧 THỬ SỬA
MIN_SECONDS = 3            # ID sống ít nhất bao nhiêu giây thì coi là ID "bền"
PEOPLE_ON_COURT = 10       # số người thật trên sân (tự đếm trong video; có tính trọng tài không?)

require_file(TRACKS_CSV, "01_track_video.py")
meta = load_json(META_JSON)
fps = meta["fps"]
tracks = read_tracks(TRACKS_CSV)

# ------------------------------------------------------------------
# Bước 1: Với mỗi ID, tìm frame đầu, frame cuối và số frame xuất hiện
# ------------------------------------------------------------------
lifetimes = {}             # ID → {"first": ..., "last": ..., "count": ...}
for row in tracks:
    track_id, frame = row["id"], row["frame"]
    if track_id not in lifetimes:
        lifetimes[track_id] = {"first": frame, "last": frame, "count": 0}
    info = lifetimes[track_id]
    info["first"] = min(info["first"], frame)
    info["last"] = max(info["last"], frame)
    info["count"] += 1

n_ids = len(lifetimes)
durable_ids = [tid for tid, info in lifetimes.items() if info["count"] / fps >= MIN_SECONDS]

print(f"Tracker: {meta['tracker']}   |   {meta['frames_processed']} frame ≈ {meta['frames_processed'] / fps:.1f} giây")
print(f"\nTổng số ID khác nhau:            {n_ids}")
print(f"Số ID bền (sống ≥ {MIN_SECONDS} giây):       {len(durable_ids)}")
print(f"Độ phân mảnh = {n_ids} ID / {PEOPLE_ON_COURT} người = {n_ids / PEOPLE_ON_COURT:.1f} lần")

print("\n15 ID sống lâu nhất:")
print("  ID | frame đầu | frame cuối | số giây có mặt")
print("-----+-----------+------------+---------------")
sorted_ids = sorted(lifetimes, key=lambda tid: lifetimes[tid]["count"], reverse=True)
for track_id in sorted_ids[:15]:
    info = lifetimes[track_id]
    print(f" {track_id:3d} | {info['first']:9d} | {info['last']:10d} | {info['count'] / fps:6.1f}")

# ------------------------------------------------------------------
# Bước 2: Số người mỗi frame, và những lúc số người TỤT đột ngột
# ------------------------------------------------------------------
# Nếu từ frame này sang frame sau tự nhiên mất ≥ 2 người, nhiều khả năng
# là có người bị che (chứ cầu thủ không thể biến mất khỏi sân trong 1/25 giây).
by_frame = group_by_frame(tracks)
frames = list(range(meta["frames_processed"]))
people_per_frame = [len(by_frame.get(frame, [])) for frame in frames]

sudden_drops = []
for frame in frames[1:]:
    if people_per_frame[frame] - people_per_frame[frame - 1] <= -2:
        sudden_drops.append(frame)

print(f"\nSố người mỗi frame: ít nhất {min(people_per_frame)}, nhiều nhất {max(people_per_frame)}, "
      f"trung bình {sum(people_per_frame) / len(people_per_frame):.1f}")
print(f"Có {len(sudden_drops)} lần số người tụt ≥ 2 so với frame trước. Vài frame đầu tiên: {sudden_drops[:10]}")

# ------------------------------------------------------------------
# Bước 3: Vẽ biểu đồ "tuổi thọ" của ID
# ------------------------------------------------------------------
# Hai biểu đồ xếp chồng, dùng chung trục thời gian (sharex=True)
fig, (ax_ids, ax_count) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})

# Sắp xếp ID theo lúc xuất hiện, để biểu đồ đọc từ trên xuống như dòng thời gian
ids_by_start = sorted(lifetimes, key=lambda tid: lifetimes[tid]["first"])
for row_number, track_id in enumerate(ids_by_start):
    info = lifetimes[track_id]
    start_s, end_s = info["first"] / fps, info["last"] / fps
    color = "tab:blue" if track_id in durable_ids else "lightgray"
    ax_ids.plot([start_s, end_s], [row_number, row_number], color=color, linewidth=3)

ax_ids.set_ylabel("ID (xếp theo lúc xuất hiện)")
ax_ids.set_yticks([])
ax_ids.set_title(f"Mỗi vạch là một ID  —  xanh: sống ≥ {MIN_SECONDS} s ({len(durable_ids)} ID), "
                 f"xám: ID ngắn ({n_ids - len(durable_ids)} ID)")

seconds = [frame / fps for frame in frames]
ax_count.plot(seconds, people_per_frame, color="tab:orange")
ax_count.axhline(PEOPLE_ON_COURT, color="black", linestyle="--", linewidth=1, label=f"{PEOPLE_ON_COURT} người thật")
ax_count.set_xlabel("Thời gian (giây)")
ax_count.set_ylabel("Số người")
ax_count.legend()

chart_path = OUTPUT_DIR / "02_id_timeline.png"
fig.tight_layout()
fig.savefig(chart_path, dpi=120)
plt.close(fig)

print(f"\n✅ Biểu đồ: {chart_path}")
print("\nCâu hỏi:")
print("  1. Có bao nhiêu vạch xám ngắn? Chúng thường bắt đầu ngay sau lúc một vạch xanh kết thúc không?")
print("  2. Chạy lại bài 01 với TRACKER = \"botsort.yaml\" rồi chạy lại bài này. Độ phân mảnh tốt hơn hay tệ hơn?")
print("  3. Thử my_bytetrack.yaml với track_buffer = 90. Số ID thay đổi thế nào?")
