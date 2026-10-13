# Lab 25 — Báo cáo ngắn: NimbusAI cắt 45,8% hóa đơn GPU

**Học viên:** Phùng Thành An — 2A202603006 · **Repo:** https://github.com/Mashall-Anie/K4-L3-DAY25-PhungThanhAn-2A202603006-GpuFinOps
**Nguồn số liệu:** `outputs/report.md`, `outputs/savings.png`, `outputs/focus_export.csv` (sinh tất định, seed 25)

**Kết quả:** *baseline $27,133 → optimized $14,705, tiết kiệm 45,8%, `$/1M-token` 6.488 → 1.126.*

---

## 1. Baseline vs. Optimized

| Chỉ số | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| Chi phí/tháng | $27,133 | $14,705 | −$12,428 (−45,8%) |
| `$/1M-token` (inference API) | $6.488 | $1.126 | −82,6% |
| `$/1M-token` (all-in, gồm thuê GPU) | $120.06 | $65.07 | −45,8% |

Mức 45,8% nằm trong khoảng mục tiêu 40–95% của lab. All-in cao hơn nhiều so với bill API vì nó gánh cả tiền thuê GPU training — đây là con số nên đưa cho ban điều hành, còn `$/1M-token` API dùng để so giữa các phương án phục vụ cùng một lượng token.

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | Tiết kiệm/tháng | % tổng |
|---|---|---|
| Purchasing (spot/reserved) | $10,105 | 81% |
| Inference (cascade/cache/batch) | $1,212 | 10% |
| Kill idle GPUs | $600 | 5% |
| Right-size over-provisioned GPUs | $511 | 4% |

**Vì sao purchasing chiếm tới 81%?** Vì đây là chỗ duy nhất **đơn giá** thay đổi ($25,667 → $15,562/tháng = −39,4%), còn ba đòn bẩy kia chỉ tối ưu *cách dùng* tài nguyên đã mua. Các chiết khấu **nhân** chứ không cộng: batch ×0,5 × cache ×0,1 = 0,05 — nên cascade + cache + batch cho 82,6% chứ không phải 155%.

## 3. GPU-Util Lie — cơ chế và tác động tài chính

**Cơ chế:** `nvidia-smi` chỉ đo **thời gian clock ở trạng thái bận**, không đo FLOPs thực sự sinh ra. Một kernel chờ dữ liệu từ HBM vẫn giữ clock bận và vẫn hiện util ≈ 100% — nên GPU-Util **không dùng được để ra quyết định mua sắm**; phải đo MFU (FLOPs) và MBU (băng thông).

- `gpu-h100-4`: **util 98,2% nhưng MFU 0,194 / MBU 0,207** → trả giá H100 ($2.50/giờ) để nhận 225,5 TFLOPS (peak 990) và 0,908 TB/s (peak 3,35).
- Đối chứng: 4 GPU H100 khác (gpu-h100-0…3) đạt MFU 0,40–0,43 → 0,19 **không phải workload khó**, mà là mua sai card.
- `gpu-a10g-1` cũng "lie" (96,9% / MFU 0,268) nhưng **không hạ cấp được** vì đã là card rẻ nhất đủ dùng → minh chứng cho việc không hạ cấp mù quáng.
- **Tác động:** $511/tháng (right-size H100→A100) + 216 kWh/tháng (700W → 400W) + $600/tháng từ 8 giờ/ngày GPU bỏ không của `gpu-h100-5`.

## 4. Phần mở rộng đã làm (5/5, đều có số đo trước/sau)

| # | Extension | Số đo | Insight |
|---|---|---|---|
| 1 | Policy mua theo rủi ro (`tier_decision`) | từ chối **$2,083/tháng** phantom savings; rework **$451/tháng**; linh hoạt 1yr thay 3yr tốn thêm **$2,232/tháng** | Phantom có **2 chiều**: job ngắn thì ảo đến từ reserved, service 24/7 thì ảo đến từ spot |
| 2 | Right-size theo MBU (`rightsize_gpu`) | H100→A100: **$511/tháng**, −216 kWh | Chỉ hạ cấp khi lệch **cả hai** trần (MFU<0,30 **và** MBU<0,60); 5 GPU bị flag nhưng chỉ 1 hạ được |
| 3 | `cache_is_worth_it()` | reuse **8,72 lần/prefix** vs break-even **1,28**; chiết khấu thật **76,8%** (không phải 90%, haircut 14,6%); **19,3% session** không nên bật cache | Cache là **khoản vay**: chữ 1,25× phải trả bằng lượt đọc |
| 4 | Ngân sách reasoning | 8,4% request nhưng **35,6% output token**, **94% năng lượng** inference (148 Wh vs 0,86 Wh/query); cap rule tiết kiệm **379,9 kWh/tháng** | Reasoning là bài toán **năng lượng**, không phải bài toán $ |
| 5 | Carbon-aware scheduling | 5 job → `us-east-wa`: **−76% CO₂e** (680→161 kg), **−$116/tháng** tiền điện trên 1.789 kWh | Điện sạch và điện rẻ **trùng nhau**; `europe-north1` sạch nhất (−92%) nhưng tiết kiệm tiền ít hơn ($54) |

## 5. Khuyến nghị cho NimbusAI — 3 việc có thứ tự

1. **Chốt chiến lược mua GPU trước ($10,105/tháng)** — lớn nhất, ít rủi ro nhất, không cần đổi code sản phẩm. Reserved chỉ cho service đo được duty ≥ break-even và vòng đời dài hơn kỳ cam kết; spot + checkpoint cho mọi job ngắt được.
2. **Mặc định model nhỏ + cache có kiểm chứng + cap reasoning ($1,212/tháng và 94% năng lượng inference)** — đưa luật vào gateway trước khi traffic reasoning tăng.
3. **Sửa hình dạng fleet + chỗ đặt job ($1,111/tháng + −76% CO₂e)** — hạ cấp GPU không dùng tới trần hiệu năng, tắt idle, chuyển job interruptible sang `us-east-wa`; song song bịt 197 request thiếu tag `project` (tag coverage đạt 92% → **cổng chargeback đang mở**).

**Ghi chú độ tin cậy:** giá là snapshot tháng 6/2026; kinh tế purchasing tính theo run-rate 30 ngày; mô hình năng lượng reasoning (hệ số 80×) mang tính định hướng. Con số right-size **giảm** từ $655 (policy cũ) xuống $511 (policy có chứng minh) — nhỏ hơn nhưng trung thực hơn.

**Kiểm chứng:** `python verify.py` → **11/11 checks passed** · `pytest -q` → **15 passed** · `outputs/report.md` + `outputs/savings.png` + `outputs/focus_export.csv` (2.400 dòng FOCUS, đủ `team`/`project`).