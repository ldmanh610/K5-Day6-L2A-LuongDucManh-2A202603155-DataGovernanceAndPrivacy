# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Mọi số liệu lấy từ file do script sinh ra: `out/poc/run_info.json`, `out/poc/decision.json`, `out/poc/ranked_review.csv`, `out/baseline/run_info.json`.
> 3–5 trang.

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G01 |
| Challenge | T02 · Raw QC Triage — Phát hiện ảnh lỗi kỹ thuật (mờ, tối, cháy sáng) |
| Thành viên + vai trò | 1. Luong Duc Manh — 2A202603155 (Lead / Algorithm / Report / QA) |
| Git URL | https://github.com/ldmanh610/K5-Day6-L2A-LuongDucManh-2A202603155-DataGovernanceAndPrivacy |
| Tag `demo-180m` (commit) | (chạy: `git tag -a demo-180m -m "Demo freeze" && git push origin demo-180m`) |
| Tag `final-24h` (commit) | (chạy: `git tag -a final-24h -m "Final submission" && git push origin final-24h`) |
| Deadline (giờ VN) | 2026-11-15 15:00 ICT |

## 01 · Problem statement

- **Ai bị ảnh hưởng:** Kỹ sư data và model owner — nếu ảnh đầu vào bị mờ/tối/cháy sáng lọt vào tập train, chất lượng mô hình AI xe tự lái sẽ suy giảm mà không có cảnh báo.
- **Điểm nghẽn cụ thể:** Trong 96 ảnh raw thu thập từ 2 camera (cam_A, cam_B) qua 3 điều kiện ánh sáng (day, dusk, night), một phần bị lỗi kỹ thuật nhưng không có nhãn tự động → phải review toàn bộ bằng tay = tốn kém.
- **Đầu vào → đầu ra:** `fixtures/t02_raw_qc/images.csv` (96 ảnh, metadata: image_id / path / camera_id / light / split) → `out/poc/ranked_review.csv` (image_id, score, pred_defect_type, pred_action, reason) + `out/poc/contact_sheet.html` + `out/poc/decision.json`.
- **Tiêu chí GO / REWORK (đã chốt lúc 15:00 ngày 14/11/2026, trước khi xem eval):**
  - **REWORK:** tỉ lệ RECAPTURE trong top-20 ≥ 30%
  - **GO:** tỉ lệ RECAPTURE trong top-20 < 30%
  - Ngưỡng 0.30 đóng băng trong code `src/pipeline.py::decide()` trước khi nhìn kết quả eval.
- **Giả định (cần DPO/legal xác nhận):** Dữ liệu synthetic CC0 — không có PII thật; phương pháp chỉ đánh giá chất lượng kỹ thuật (không phân tích nội dung ảnh người/biển số).

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | ~58 (60% scene gốc) | `fixtures/t02_raw_qc/images.csv` trong repo | CC0 | Có |
| eval / heldout | ~38 (40% scene gốc) | `fixtures/t02_raw_qc/images.csv` trong repo | CC0 | Có |

- **Schema:** `image_id, path, camera_id (cam_A/cam_B), light (day/dusk/night), split (dev/eval)`. 96 ảnh × 4 variants = 24 scene.
- **Seed:** 42. Chia split theo scene_id (2 ký tự đầu của image_id) để tránh rò rỉ dữ liệu.
- `manifest.csv` verified sha256: lệnh `python -m tools.verify_manifests fixtures/t02_raw_qc/manifest.csv` — kết quả từ preflight: `[OK] t02 fixtures/t02_raw_qc` sau khi fix git `core.autocrlf=false`.
- **Dữ liệu thật:** Không sử dụng (giảng viên chưa upload `data/real/`).

## 03 · Research & design

**References:**
| URL | Ngày xem | Ý nghĩa |
|---|---|---|
| https://ieeexplore.ieee.org/document/843417 | 2026-11-14 | Pech-Pacheco et al. (2000) — Laplacian variance làm proxy đo độ nét ảnh; số cao = nét, số thấp = mờ |
| https://www.iso.org/standard/83083.html | 2026-11-14 | ISO/IEC TR 5259-6:2026 — QC dữ liệu AI, metric review@budget |
| https://acdc.vision.ee.ethz.ch/ | 2026-11-14 | ACDC dataset — phân loại ảnh xe theo điều kiện thời tiết/ánh sáng, tham khảo cách chia split |

**Data flow:**
```
images.csv (96 ảnh)
  → _extract_features(): Laplacian variance + brightness stats (mean, std, frac_dark, frac_saturated)
  → _compute_scores(): weighted sum (blur 45% + dark 25% + glare 20% + contrast 10%) + chuẩn hoá theo camera_id
  → rank(): top-20 → review_queue.csv + contact_sheet.html
  → review(): áp override nếu reviewer đã điền review_queue_reviewed.csv
  → decide(): REWORK nếu RECAPTURE rate ≥ 30%, GO nếu không
  → decision.json + ranked_review.csv
```

**Thành phần offline:** Chỉ dùng numpy, scipy, Pillow (không model pretrained, không network call). Kiểm tra bằng `--offline` flag — `[OK] offline guard blocks outbound network`.

**Trade-offs đã cân nhắc:**
1. Laplacian đơn giản nhưng cam_B có quang học mềm hơn cam_A → chuẩn hoá theo `camera_id` (chia cho median của từng cam).
2. `frac_dark` cao ở ảnh đêm hữu ích → risk false alarm; giải pháp: slice report theo `light`.
3. Không dùng deep learning để đảm bảo chạy offline hoàn toàn và tái hiện 100%.

## 04 · Baseline vs PoC

- **Baseline:** Random score (seed 42) — `src.examples.random_baseline`. Runtime: **1.683s / 96 ảnh**.
- **PoC:** Multi-feature scoring (Laplacian blur + brightness dark/glare + contrast, chuẩn hoá theo camera). Runtime: **20.18s / 96 ảnh** (CPU-only, feature extraction).
- Cùng budget B=20 / cùng seed 42 / cùng fixtures `t02_raw_qc`.

**Bảng 1 · Baseline vs PoC** (số liệu từ `out/*/run_info.json` và `out/poc/decision.json`):

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Primary: Defect Recall@20 | ~0.33 (random) | TBD — cần overlay GT của giảng viên | higher is better |
| Precision@20 | ~0.33 (random) | TBD | higher is better |
| RECAPTURE rate top-20 | random ~50% | 100% (toàn bộ top-20 bị xếp RECAPTURE) | Decision = REWORK |
| Runtime / 96 items | 1.683s | 20.18s | PoC chậm hơn do feature extraction ảnh |
| n_items processed | 96 | 96 | Đủ 100% fixture |

> **Ghi chú:** Điểm chính thức (Defect Recall@20) sẽ được giảng viên chấm sau khi overlay GT lên `ranked_review.csv`. Nhóm không có quyền đọc `fixtures/t02_raw_qc/gt/`.

## 05 · Evaluation protocol

- **Công thức:** Recall@B = #ảnh lỗi thật trong top-B / #ảnh lỗi thật toàn bộ eval split. Precision@B = #ảnh lỗi thật trong top-B / B.
- **Budget B = 20**, đóng băng trong `configs/tracks/t02.yaml`.
- **Threshold REWORK:** `recapture_rate >= 0.30`, đóng băng trong `src/pipeline.py::decide()` lúc 15:00 ngày 14/11/2026.
- **Độ bất định:** N~96 nên CI rộng; sử dụng Wilson CI 95% cho Recall@B và Precision@B nếu cần báo cáo thống kê.
- **Lệnh chạy chính xác:**

```bash
# Setup (1 lần)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Chạy PoC
docker run --rm -v "%cd%:/workspace" -w /workspace day6-fast:latest python -m src.main --config configs/lab.yaml --track t02 --output out/poc --run-name poc --offline --seed 42

# Chạy baseline
docker run --rm -v "%cd%:/workspace" -w /workspace day6-fast:latest python -m src.main --track t02 --pipeline src.examples.random_baseline --output out/baseline --run-name baseline --offline --seed 42

# Chạy tests
docker run --rm -v "%cd%:/workspace" -w /workspace day6-fast:latest pytest -q
```

## 06 · Result slices

| Slice | N | Decision/score | Ghi chú |
|---|---|---|---|
| light=day | ~40 | TBD (cần GT) | Ảnh ngày: blur là failure mode chính |
| light=dusk | ~16 | TBD | Ánh sáng thấp, nguy cơ dark+blur |
| light=night | ~40 | TBD | Frac_dark cao → risk false alarm |
| camera=cam_A | ~48 | TBD | Quang học cứng hơn, Laplacian cao hơn |
| camera=cam_B | ~48 | TBD | Quang học mềm hơn → đã chuẩn hoá |

**Failure case 1 — Ảnh đêm hữu ích bị xếp RECAPTURE:**
Ảnh chụp ban đêm nét nhưng `frac_dark` cao → score tổng cao → lọt top-20 RECAPTURE.
Root cause: `frac_dark` không phân biệt "tối do thiếu sáng" và "tối do môi trường ban đêm bình thường".
Khắc phục đề xuất: slice ngưỡng dark_score theo `light == night`.

**Failure case 2 — cam_B bị báo nhầm blur:**
Cam_B có quang học mềm hơn cam_A, Laplacian trung bình thấp hơn → bị báo blur sai.
Root cause: ngưỡng Laplacian chung cho cả 2 camera.
Khắc phục: **đã áp dụng** — chia Laplacian cho median của từng `camera_id`.

**Latency:** ~210ms/ảnh trên CPU WSL2 (Python 3.12, numpy/scipy/Pillow). 96 ảnh = 20.18s tổng.

## 07 · Human & decision

- **Review queue:** `out/poc/review_queue.csv` (top-20 theo score). Mở `out/poc/contact_sheet.html` bằng Chrome/Edge để xem ảnh trực quan offline.
- **Override + audit log:** Reviewer điền cột `reviewer_decision` vào `review_queue_reviewed.csv`, copy vào `out/poc/`, chạy lại `python -m src.main ...` → `out/poc/audit_log.jsonl` ghi lại mọi thay đổi.
- **Decision policy:** `decide()` trong `src/pipeline.py`:
  - REWORK (reason_code=`HIGH_DEFECT_RATE`) nếu ≥ 30% trong top-20 là RECAPTURE
  - GO (reason_code=`ACCEPTABLE_DEFECT_RATE`) nếu < 30%
  - Owner: `LuongDucManh (QC-Engineer)`
- **Kết quả thực tế:** `decision.json` → `"status": "REWORK"`, `"reason": "RECAPTURE rate in top-20 = 100.0% >= 30%"` — batch này cần chụp lại.

## 08 · Reproduce & tests

```bash
# Chạy đầy đủ trong Docker (không cần cài gì thêm trên Windows)
docker run --rm -v "C:\Users\admin\Downloads\K5-Day6-L2A-LuongDucManh-2A202603155-DataGovernanceAndPrivacy:/workspace" -w /workspace day6-fast:latest python -m src.main --config configs/lab.yaml --track t02 --output out/poc --run-name poc --offline --seed 42

docker run --rm -v "C:\Users\admin\Downloads\K5-Day6-L2A-LuongDucManh-2A202603155-DataGovernanceAndPrivacy:/workspace" -w /workspace day6-fast:latest pytest -q
```

**Output của `pytest -q`:**
```
...s....s...s............................ssssssssssssss................s [ 77%]
sssssssssssssssssssss                                                    [100%]
37 passed, 41 skipped — 0 failed
```

**Môi trường:** Docker `python:3.11-slim` / Python 3.11.18 / Linux WSL2 x86_64 / seed 42
**Git SHA (commit mới nhất):** `b3c88d8` — `test: update test_cli and verify 100% pytest pass`

## 09 · Limitations & ethics

- **Phạm vi:** Chỉ phát hiện lỗi kỹ thuật (độ nét, độ sáng); không kiểm tra nội dung (xe, người, biển số).
- **False alarms:** Ảnh đêm nét có thể bị xếp RECAPTURE do `frac_dark` cao → reviewer cần kiểm tra `contact_sheet.html`.
- **Scale 10×:** Với ~960 ảnh, cần batch processing và cache feature để giảm latency từ ~20s xuống còn vài giây.
- **Privacy:** Dữ liệu synthetic CC0 — không có PII thật. **Không tuyên bố "đạt GDPR / luật Việt Nam"** từ checklist tự động; cần DPO/legal xác nhận trước khi áp dụng vào dữ liệu thật.
- **Kế hoạch cải tiến:** (1) Thêm ROI-based scoring (vùng biển số/đầu xe); (2) Điều chỉnh ngưỡng dark_score theo `light`; (3) Thử FFT blockiness thay cho DCT 8×8.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence:**

| Học viên | Git author/email | Module / vai trò | Commit SHA | Trình bày? |
|---|---|---|---|---|
| Luong Duc Manh — 2A202603155 | wazi2752004@gmail.com | Lead + Algorithm + Report + QA + CI | 4c259ee, b3c88d8 | Yes |

Chạy để xác nhận: `git shortlog -sne --all`

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa |
|---|---|---|
| https://ieeexplore.ieee.org/document/843417 | 2026-11-14 | Pech-Pacheco et al. (2000) — Laplacian variance làm proxy đo độ nét ảnh cho autofocus |
| https://www.iso.org/standard/83083.html | 2026-11-14 | ISO/IEC TR 5259-6:2026 — Tiêu chuẩn QC dữ liệu AI, metric review@budget |
| https://acdc.vision.ee.ethz.ch/ | 2026-11-14 | ACDC dataset — phân loại ảnh xe theo điều kiện thời tiết/ánh sáng, tham khảo split strategy |

---

**Bảng 3 · Checklist chốt khi nộp**

- [x] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật (synthetic CC0).
- [x] Đúng 1 lệnh run, 1 lệnh tests; không cần download khi chạy (Docker image `day6-fast:latest`).
- [x] Có baseline (random) + PoC (multi-feature), ≥1 failure case, reviewer evidence (contact_sheet.html), decision log (decision.json).
- [x] Có license/source provenance (CC0), data splits (dev/eval by scene), seed 42, offline environment.
- [x] Thành viên có commit thực chất (4c259ee, b3c88d8) và được nêu minh bạch trong bảng trên.
- [ ] Overlay GT từ giảng viên để lấy Defect Recall@20 chính thức.
- [ ] Tag `demo-180m` và `final-24h` sau khi nộp.
