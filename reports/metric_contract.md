# Metric contract — T02 Raw QC Triage
# (Chốt lúc 15:00 ngày 14/11/2026, trước khi xem kết quả eval)

| Mục | Khai báo của nhóm |
|---|---|
| Track | T02 — Raw QC Triage |
| Primary metric | Defect Recall@B (B=20) |
| Secondary metric | Precision@B (B=20) |
| Tăng tốt hay giảm tốt? | Higher is better |
| Đơn vị đánh giá | image |
| Công thức Recall@B | #ảnh lỗi thật trong top-B / #ảnh lỗi thật toàn bộ eval split |
| Công thức Precision@B | #ảnh lỗi thật trong top-B / B |
| Cách ghép prediction ↔ GT | Khớp theo `image_id`; GT chưa lộ khi chạy; tự kiểm bằng dev split |
| Candidate pool cố định | Toàn bộ 96 ảnh (`fixtures/t02_raw_qc/images.csv`), seed 42 |
| Budget B | 20 |
| Seed | 42 |
| Threshold REWORK / GO | REWORK nếu RECAPTURE rate trong top-20 ≥ 0.30; đóng băng trong `src/pipeline.py::decide()` |
| Eval split (không dùng để tune) | `split == eval` (~40 ảnh, 40% scene gốc) |
| Chỉ số lỗi xấu | Miss rate (ảnh lỗi bị xếp KEEP) và False alarm (ảnh tốt bị RECAPTURE) |
| Slices sẽ báo cáo (≥2) | Theo `light` (day/dusk/night) và theo `camera_id` (cam_A/cam_B) |
| Độ bất định | Wilson CI 95% cho Recall@B và Precision@B; N~96 nên CI rộng |
| Ngày giờ chốt + người chốt | 2026-11-14 15:00 ICT — LuongDucManh (2A202603155) |
