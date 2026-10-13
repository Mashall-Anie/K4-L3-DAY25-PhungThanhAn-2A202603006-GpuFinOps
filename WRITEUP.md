# Lab 25 — Báo cáo ngắn: NimbusAI cắt 45,8% hóa đơn GPU

**Học viên:** Phùng Thanh An — 2A202603006 · **Repo:** K4-L3-DAY25-PhungThanhAn-2A202603006-GpuFinOps
**Nguồn số liệu:** `outputs/report.md`, `outputs/savings.png`, `outputs/focus_export.csv` (sinh tất định, seed 25)

---

## 1. Baseline vs. Optimized

| Chỉ số | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| Chi phí GPU/tháng | **$27,133** | **$14,705** | **−$12,428 (−45,8%)** |
| `$/1M-token` (bill inference) | **$6.488** | **$1.126** | **−82,6%** |
| `$/1M-token` (all-in, gồm thuê GPU) | $120.06 | $65.07 | −45,8% |
| Token phục vụ | 7.53M token/ngày (226M/tháng) | không đổi | — |

Điểm cần nhấn: cùng một lượng token, hóa đơn giảm gần một nửa; riêng phần inference API thì rẻ đi **hơn 5,7 lần**. Con số `$/1M-token` toàn phần (120 → 65 USD) cao hơn nhiều so với bill API vì nó gánh cả tiền thuê GPU training — đây mới là con số nên đưa cho ban điều hành, còn `$/1M-token` API dùng để so giữa các phương án phục vụ.

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | Tiết kiệm/tháng | % tổng | Vì sao |
|---|---|---|---|
| **Purchasing (spot/reserved)** | **$10,105** | **81%** | Đòn bẩy lớn nhất |
| Inference (cascade/cache/batch) | $1,212 | 10% | $6.488 → $1.126 /1M-token |
| Kill idle GPUs | $600 | 5% | 8 h/ngày GPU nằm không |
| Right-size GPU quá khổ | $511 | 4% | H100 → A100 |
| **Tổng** | **$12,428** | **100%** | |

**Vì sao purchasing chiếm 81%?** Vì đây là chỗ duy nhất mà giá đơn vị (không phải hiệu suất) thay đổi: $25,667/tháng theo on-demand → $15,562/tháng khi đúng cấu hình mua (−39,4%), trong khi mọi đòn bẩy khác chỉ tối ưu *cách dùng* tài nguyên đã mua. Cụ thể: 3 service inference chạy 24/7 (A10G/A100/L4) rơi vào reserved 3 năm (duty cycle 100% ≥ break-even 56–60%), còn 5 job training/batch có thể bị ngắt thì đi spot — với rủi ro rework chỉ **351 GPU-giờ ($451/tháng)**.

Hai phát hiện đáng chú ý từ policy mới (Extension 1): (a) nó **từ chối $2,083/tháng "phantom savings"** — phần tiết kiệm mà cách chọn tier ngây thơ theo giá/giờ sẽ claim được bằng cách cam kết 3 năm cho job chỉ sống 3–30 ngày; (b) trả giá cho tính linh hoạt: chọn reserved 1 năm thay vì 3 năm cho 3 service persistent tốn thêm $2,232/tháng.

**Inference** rẻ đi 82,6% nhờ 3 đòn bẩy nhân nhau: cascade sang model nhỏ (rẻ 15×), cache input (10% giá), batch API (−50%). Vì các discount **nhân** chứ không cộng, batch + cache 100% chỉ còn 0,05 hóa đơn gốc.

## 3. GPU-Util Lie

`nvidia-smi` đo "clock đang bận", không đo FLOPs sinh ra. Trong 11 GPU của telemetry:

| GPU | GPU-Util | MFU | MBU | Diễn giải |
|---|---|---|---|---|
| **gpu-h100-4** | **98,2%** | **0,194** | 0,207 | Báo bận gần 100%, thực nhận ~19% FLOPs và ~21% băng thông |
| gpu-a10g-1 | 96,9% | 0,268 | 0,302 | Lie thứ hai, nhưng đã là card rẻ nhất đủ dùng → không thể hạ cấp |

**Tác động tài chính:** card này ngốn một giờ H100 đầy giá $2.50 để trả về 225,5 TFLOPS (peak 990) và 0,908 TB/s (peak 3,35). Vì workload chỉ cần hơn 1,59 TB/s ở 60% MBU, một **A100 80GB ($1.79/giờ)** làm được đúng việc đó → **$511/tháng**, và vì A100 chỉ ăn 400W thay vì 700W nên còn giảm **216 kWh/tháng**. Cộng thêm 8 h/ngày GPU bỏ không (gpu-h100-5, $600/tháng) và phần do utilization giả định sai, tổng thiệt hại từ "GPU bận nhưng vô dụng" là **$1,111/tháng**. Bằng chứng đối chứng: 4 GPU H100 còn lại có MFU 0,40–0,43 — đó là mức dùng bình thường, nên 0,19 không phải "workload khó" mà là "mua sai card".

## 4. Phần mở rộng đã làm (5/5)

| # | Extension | Kết quả đo được | Insight quan trọng nhất |
|---|---|---|---|
| 1 | `pricing.tier_decision()` — policy mua GPU theo rủi ro | $15,627 → $15,562/tháng; từ chối **$2,083/tháng** phantom savings; rework $451/tháng; trả thêm $2,232/tháng nếu chọn 1 năm thay vì 3 năm | So sánh giá/giờ mà **không kiểm tra vòng đời job** sẽ tạo ra tiết kiệm không bao giờ tồn tại |
| 2 | Right-size theo MBU (`metrics.rightsize_gpu`) | gpu-h100-4 H100→A100: **$511/tháng**, −216 kWh/tháng, không đổi workload | Chỉ hạ cấp khi GPU lệch **cả** hai trần (MFU < 0,30 và MBU < 0,60); card đang dùng 55% băng thông thì hạ cấp chỉ làm job chậm đi |
| 3 | `pricing.cache_is_worth_it()` | Reuse thực đo **8,72 lần gửi/prefix** so với break-even **1,28**; chiết khấu thật **76,8%** chứ không phải 90% (haircut 14,6%); **57 session (19,3%)** chỉ gửi 1 lần → phải tắt cache; giá trị cache đã audit $27/tháng | Cache là một **khoản vay**: chữ ghi 1,25× phải được trả lại bằng lượt đọc, nên "90% off" hầu như không bao giờ là số thật |
| 4 | Ngân sách reasoning | 8,4% request nhưng **35,6% output token** và **94,0% năng lượng** inference (148 Wh/query so với 0,86 Wh); luật "reasoning chỉ ở tier lớn, còn lại cap 641 token" tiết kiệm $3,17/tháng + **379,9 kWh/tháng** | Reasoning là bài toán **năng lượng**, không phải bài toán $ — nó gần như vô hình trên hóa đơn token nhưng chiếm gần hết điện năng |
| 5 | Carbon-aware scheduling | Chuyển 5 job interruptible us-east-1 → us-east-wa: **680 → 161 kgCO₂e (−76%)** và **−$116/tháng** tiền điện trên 1.789 kWh | Ở đây điện rẻ và điện sạch **trùng nhau**: quyết định vì carbon cũng là quyết định vì tiền (`europe-north1` sạch nhất 30 gCO₂e/kWh, `us-east-wa` thắng khi cân cả giá) |

## 5. Khuyến nghị cho NimbusAI (3 việc đầu tiên nếu tôi là FinOps lead)

1. **Chốt lại chiến lược mua GPU trước ($10,105/tháng).** Áp policy risk-adjusted: reserved cho service có duty cycle đo được ≥ break-even và vòng đời dài hơn kỳ cam kết; spot + checkpoint cho mọi job có thể ngắt; **không bao giờ** cam kết dài hơn vòng đời workload. Đây là đòn bẩy lớn nhất, thấp rủi ro nhất và không cần đổi code sản phẩm.
2. **Mặc định model nhỏ + cache có kiểm chứng ($1,212/tháng, và giảm 94% năng lượng reasoning).** Cascade theo độ khó, batch các request chờ được, chỉ bật cache cho prefix được đọc ≥ 2 lần — đồng thời đưa luật cap reasoning vào gateway trước khi traffic tăng, vì đó là nơi năng lượng bị đốt.
3. **Sửa hình dạng fleet và chỗ đặt job ($1,111/tháng + 76% CO₂e).** Hạ cấp GPU không dùng tới trần hiệu năng, tắt 8 giờ idle/ngày, chuyển job interruptible sang vùng sạch–rẻ. Song song: bịt 197 request thiếu tag `project` (8% spend) trước khi bật chargeback, và dùng `focus_export.csv` làm nguồn dữ liệu chung với gate tag coverage 80% — hiện đã đạt 92%, **cổng chargeback đang mở**.

> **Ghi chú độ tin cậy:** giá là snapshot tháng 6/2026; kinh tế purchasing tính theo run-rate 30 ngày cho mỗi job; mô hình năng lượng reasoning (hệ số 80×) mang tính định hướng — ở quy mô traffic này nó cho năng lượng lớn hơn cửa sổ telemetry 11 GPU, nên đọc Wh như số tương đối. Toàn bộ pipeline chạy lại được bằng `python data/generate.py && python missions/run_all.py` với `python verify.py` (11/11) và `pytest -q` (15 passed).

---

*Checklist nộp bài:* `outputs/report.md` ✅ · `outputs/savings.png` ✅ · `outputs/focus_export.csv` ✅ (2.400 dòng FOCUS) · bài viết ngắn ✅ · ≥2 extension ✅ (đã làm 5/5, có số đo trước/sau).
