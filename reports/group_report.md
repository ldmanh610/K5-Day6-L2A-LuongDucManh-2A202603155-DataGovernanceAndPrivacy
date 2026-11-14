# Group Report - Day 06 Data Lab (Field-to-Release PoC)

## 00 - Header

| Muc | Gia tri |
|---|---|
| Group ID | G01 |
| Challenge | T02 - Raw QC Triage (Phat hien anh loi ky thuat: mo, toi, chay sang) |
| Thanh vien + vai tro | 1. Luong Duc Manh - 2A202603155 (Lead / Algorithm / Report / QA) |
| Git URL | https://github.com/ldmanh610/K5-Day6-L2A-LuongDucManh-2A202603155-DataGovernanceAndPrivacy |
| Tag demo-180m | git tag -a demo-180m -m "Demo freeze" && git push origin demo-180m |
| Tag final-24h | git tag -a final-24h -m "Final submission" && git push origin final-24h |
| Deadline | 2026-11-15 15:00 ICT |

## 01 - Problem statement

- **Ai bi anh huong:** Ky su data va model owner -- neu anh dau vao bi mo/toi/chay sang lot vao tap train, chat luong mo hinh AI xe tu lai se suy giam ma khong co canh bao.
- **Diem nghen cu the:** Trong 96 anh raw tu 2 camera (cam_A, cam_B) qua 3 dieu kien anh sang (day, dusk, night), mot phan bi loi ky thuat nhung khong co nhan tu dong -> phai review toan bo bang tay = ton kem.
- **Dau vao -> dau ra:** `fixtures/t02_raw_qc/images.csv` (96 anh, metadata: image_id / path / camera_id / light / split) -> `out/poc/ranked_review.csv` (image_id, score, pred_defect_type, pred_action, reason) + `out/poc/contact_sheet.html` + `out/poc/decision.json`.
- **Tieu chi REWORK / GO (chot 15:00 ngay 14/11/2026, truoc khi xem eval):**
  - REWORK: ti le RECAPTURE trong top-20 >= 30%
  - GO: ti le RECAPTURE trong top-20 < 30%
  - Nguong 0.30 dong bang trong code `src/pipeline.py::decide()`.
- **Gia dinh (can DPO/legal xac nhan):** Du lieu synthetic CC0 -- khong co PII that; phuong phap chi danh gia chat luong ky thuat anh, khong phan tich noi dung.

## 02 - Data & provenance

| Split | N | Nguon | License | Synthetic? |
|---|---|---|---|---|
| dev | ~58 (60% scene goc) | fixtures/t02_raw_qc/images.csv | CC0 | Co |
| eval / heldout | ~38 (40% scene goc) | fixtures/t02_raw_qc/images.csv | CC0 | Co |

- **Schema:** `image_id, path, camera_id (cam_A/cam_B), light (day/dusk/night), split (dev/eval)`. 96 anh, 24 scene x 4 variants.
- **Seed:** 42. Chia split theo scene_id (2 ky tu dau cua image_id) de tranh ro ri du lieu.
- `manifest.csv` verified sha256: `python -m tools.verify_manifests fixtures/t02_raw_qc/manifest.csv`
- **Du lieu that:** Khong su dung (giang vien chua upload data/real/).

## 03 - Research & design

| URL | Ngay xem | Y nghia |
|---|---|---|
| https://ieeexplore.ieee.org/document/843417 | 2026-11-14 | Pech-Pacheco et al. (2000) -- Laplacian variance lam proxy do do net anh |
| https://www.iso.org/standard/83083.html | 2026-11-14 | ISO/IEC TR 5259-6:2026 -- QC du lieu AI, metric review@budget |
| https://acdc.vision.ee.ethz.ch/ | 2026-11-14 | ACDC dataset -- phan loai anh xe theo dieu kien thoi tiet/anh sang |

**Data flow:**
```
images.csv (96 anh)
  -> _extract_features(): Laplacian variance + brightness stats (mean, std, frac_dark, frac_saturated)
  -> _compute_scores(): weighted sum (blur 45% + dark 25% + glare 20% + contrast 10%)
                        + chuan hoa theo camera_id
  -> rank(): top-20 -> review_queue.csv + contact_sheet.html
  -> review(): ap override neu reviewer da dien review_queue_reviewed.csv
  -> decide(): REWORK neu RECAPTURE rate >= 30%, GO neu khong
  -> decision.json + ranked_review.csv
```

**Thanh phan offline:** Chi dung numpy, scipy, Pillow (khong model pretrained, khong network).

**Trade-offs:**
1. Laplacian don gian nhung cam_B co quang hoc mem hon -> chuan hoa theo camera_id.
2. frac_dark cao o anh dem -> risk false alarm; giai phap: slice report theo `light`.
3. Khong dung deep learning de dam bao chay offline hoan toan.

## 04 - Baseline vs PoC

- **Baseline:** Random score (seed 42) -- `src.examples.random_baseline`. Runtime: 1.683s / 96 anh.
- **PoC:** Multi-feature scoring (Laplacian + brightness dark/glare + contrast, chuan hoa theo camera). Runtime: 20.18s / 96 anh.
- Cung budget B=20 / cung seed 42 / cung fixtures t02_raw_qc.

**Bang 1 - Baseline vs PoC** (tu out/*/run_info.json va out/poc/decision.json):

| Metric | Baseline | PoC | Ghi chu |
|---|---|---|---|
| Defect Recall@20 | ~0.33 (random) | TBD (can overlay GT cua giang vien) | higher is better |
| Precision@20 | ~0.33 (random) | TBD | higher is better |
| RECAPTURE rate top-20 | ~50% (random) | 100% | Decision = REWORK |
| Runtime / 96 items | 1.683s | 20.18s | PoC cham hon do feature extraction |
| n_items processed | 96 | 96 | Du 100% fixture |

## 05 - Evaluation protocol

- **Cong thuc:** Recall@B = #anh loi that trong top-B / #anh loi that toan bo eval split
- **Budget B = 20**, dong bang trong `configs/tracks/t02.yaml`
- **Threshold REWORK:** `recapture_rate >= 0.30`, dong bang trong `src/pipeline.py::decide()`
- **Lenh chay:**
```bash
docker run --rm -v "%cd%:/workspace" -w /workspace day6-fast:latest \
  python -m src.main --config configs/lab.yaml --track t02 --output out/poc --run-name poc --offline --seed 42
```

## 06 - Result slices

| Slice | N | Ghi chu |
|---|---|---|
| light=day | ~40 | Anh ngay: blur la failure mode chinh |
| light=dusk | ~16 | Anh hoang hon: nguy co dark+blur |
| light=night | ~40 | frac_dark cao -> risk false alarm |
| camera=cam_A | ~48 | Quang hoc cung hon |
| camera=cam_B | ~48 | Quang hoc mem hon -> da chuan hoa |

**Failure case 1 -- Anh dem net bi xep RECAPTURE:**
Anh chup ban dem net nhung frac_dark cao -> score cao -> lot top-20 RECAPTURE.
Root cause: frac_dark khong phan biet "toi do thieu sang" va "toi do moi truong dem".
Khac phuc: slice nguong dark_score theo `light == night`.

**Failure case 2 -- cam_B bi bao nham blur:**
Cam_B co quang hoc mem hon cam_A, Laplacian trung binh thap hon -> bi bao blur sai.
Root cause: nguong Laplacian chung cho ca 2 camera.
Khac phuc: **Da ap dung** -- chia Laplacian cho median cua tung camera_id.

**Latency:** ~210ms/anh tren CPU WSL2 (Python 3.12). 96 anh = 20.18s tong.

## 07 - Human & decision

- **Review queue:** `out/poc/review_queue.csv` (top-20 theo score). Mo `out/poc/contact_sheet.html` bang Chrome/Edge.
- **Override + audit:** Reviewer dien `reviewer_decision` vao `review_queue_reviewed.csv`, copy vao `out/poc/`, chay lai pipeline -> `out/poc/audit_log.jsonl`.
- **Decision policy:** REWORK (reason_code=HIGH_DEFECT_RATE) neu >= 30% top-20 la RECAPTURE; GO neu khong. Owner: LuongDucManh (QC-Engineer).
- **Ket qua thuc te:** decision.json -> status="REWORK", reason="RECAPTURE rate in top-20 = 100.0% >= 30%".

## 08 - Reproduce & tests

```bash
# Chay PoC
docker run --rm -v "C:\Users\admin\Downloads\K5-Day6-L2A-LuongDucManh-2A202603155-DataGovernanceAndPrivacy:/workspace" ^
  -w /workspace day6-fast:latest ^
  python -m src.main --config configs/lab.yaml --track t02 --output out/poc --run-name poc --offline --seed 42

# Chay baseline
docker run --rm -v "C:\Users\admin\Downloads\K5-Day6-L2A-LuongDucManh-2A202603155-DataGovernanceAndPrivacy:/workspace" ^
  -w /workspace day6-fast:latest ^
  python -m src.main --track t02 --pipeline src.examples.random_baseline --output out/baseline --run-name baseline --offline --seed 42

# Chay tests
docker run --rm -v "C:\Users\admin\Downloads\K5-Day6-L2A-LuongDucManh-2A202603155-DataGovernanceAndPrivacy:/workspace" ^
  -w /workspace day6-fast:latest pytest -q
```

**Output cua pytest -q:**
```
...s....s...s............................ssssssssssssss................s [ 77%]
sssssssssssssssssssss                                                    [100%]
37 passed, 41 skipped -- 0 failed
```

**Moi truong:** Docker python:3.11-slim / Python 3.11.18 / Linux WSL2 / seed 42
**Git SHA:** b3c88d8 (latest commit)

## 09 - Limitations & ethics

- **Pham vi:** Chi phat hien loi ky thuat; khong kiem tra noi dung (xe, nguoi, bien so).
- **False alarms:** Anh dem net co the bi xep RECAPTURE vi frac_dark cao -> reviewer can kiem tra contact_sheet.html.
- **Scale 10x:** Can batch processing va cache feature de giam latency.
- **Privacy:** Du lieu synthetic CC0, khong co PII that. KHONG tuyen bo "dat GDPR / luat Viet Nam" tu checklist tu dong; can DPO/legal xac nhan truoc khi ap dung vao du lieu that.
- **Ke hoach cai tien:** (1) ROI-based scoring (vung bien so/dau xe); (2) Dieu chinh nguong dark theo light; (3) Thu FFT blockiness.

## 10 - Team contribution

| Hoc vien | Git email | Module / vai tro | Commit SHA | Trinh bay? |
|---|---|---|---|---|
| Luong Duc Manh - 2A202603155 | 369ldmanh@gmail.com | Lead + Algorithm + Report + QA | 4c259ee, b3c88d8, 454d77b | Yes |

## 11 - References

| URL | Ngay xem | Y nghia |
|---|---|---|
| https://ieeexplore.ieee.org/document/843417 | 2026-11-14 | Pech-Pacheco et al. (2000) -- Laplacian variance proxy do net anh |
| https://www.iso.org/standard/83083.html | 2026-11-14 | ISO/IEC TR 5259-6:2026 -- Tieu chuan QC du lieu AI |
| https://acdc.vision.ee.ethz.ch/ | 2026-11-14 | ACDC dataset -- phan loai anh xe theo thoi tiet |

---

**Bang 3 - Checklist chot khi nop**

- [x] Link repo va tag cuoi cung; input mau khong chua PII that (synthetic CC0).
- [x] Dung 1 lenh run, 1 lenh tests; khong can download khi chay (Docker image day6-fast:latest).
- [x] Co baseline (random) + PoC (multi-feature), >=1 failure case, reviewer evidence (contact_sheet.html), decision log (decision.json).
- [x] Co license/source provenance (CC0), data splits (dev/eval by scene), seed 42, offline environment.
- [x] Thanh vien co commit thuc chat (4c259ee, b3c88d8, 454d77b) va duoc neu minh bach.
- [ ] Overlay GT tu giang vien de lay Defect Recall@20 chinh thuc.
- [ ] Tag demo-180m va final-24h sau khi nop.
