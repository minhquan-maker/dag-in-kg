# kgqa: trả lời truy vấn logic trên FB15k-237, truy vấn viết như DAG

GQE, Query2Box, BetaE viết lại từ đầu trên PyTorch, chạy được trên CPU. Ý tưởng: mỗi dạng truy vấn là một DAG nhỏ (chiếu, giao, hợp, phủ định), một bộ thực thi duyệt DAG theo thứ tự topo và gọi các phép của mô hình.

Dữ liệu: FB15k-237 (14.505 thực thể, 237 quan hệ, thêm chiều ngược thành 474), tải từ [DeepGraphLearning/KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). Cách tách easy / hard answers theo BetaE (Ren & Leskovec, NeurIPS 2020).

## Cấu trúc

| File | Việc |
|---|---|
| `kg.py` | Đọc dữ liệu, đồ thị CSR, phép chiếu tập thực thể bằng numpy vector hóa |
| `dagq.py` | 14 dạng truy vấn viết như chương trình DAG, tính đáp án chính xác, bộ thực thi cho mô hình |
| `gen_queries.py` | Sinh truy vấn: chọn một thực thể đáp án rồi đi ngược qua DAG; với valid/test ép một cạnh held-out vào đường đi để có hard answers |
| `models.py` | GQE (điểm), Query2Box (hộp), BetaE (phân phối Beta, có phủ định) |
| `train.py` | Huấn luyện + đánh giá MRR / Hits@k trên hard answers (filtered) |
| `summarize.py` | Gom `results/*.json` thành bảng markdown |
| `run_all.sh` | Chạy toàn bộ |

## Chạy

```bash
pip install -r requirements.txt
bash run_all.sh            # 4 mô hình song song, mỗi mô hình 1 luồng CPU
STEPS_GQE=12000 STEPS_Q2B=8000 STEPS_BETAE=6000 bash run_all.sh   # bản rút gọn, khoảng 30 phút
```

## Khác với bản tham khảo (fork KGReasoning của Trần Bình)

- Truy vấn là DAG khai báo trong `dagq.py`, không phải tuple lồng nhau; thêm mô hình mới chỉ cần cài `anchor / project / intersect / negate` và hai hàm khoảng cách.
- 40.000 truy vấn train mỗi dạng (bản tham khảo 20.000), 1.000 truy vấn test mỗi dạng.
- Thêm 5 dạng có phủ định (2in, 3in, inp, pin, pni) cho BetaE.
- Negative dùng chung trong batch và KL của BetaE tính bằng nhân ma trận, nên chấm điểm toàn bộ thực thể nhanh trên CPU.
- Cosine learning rate có warm-up; số bước theo tốc độ từng mô hình.

## Kết quả

CPU 4 nhân, 1 seed, bản rút gọn: GQE 12.000 bước, Query2Box 8.000, BetaE 6.000, batch 128, 64 negative dùng chung, dim 256, γ = 24, Adam lr 1e-3. Tập test 1.000 truy vấn mỗi dạng, chỉ tính hard answers (filtered).

### MRR (%) on hard test answers

| Model | 1p | 2p | 3p | 2i | 3i | ip | pi | 2u | up | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| GQE | 11.5 | 10.2 | 9.1 | 15.1 | 14.9 | 14.9 | 14.5 | 5.3 | 4.8 | **11.1** |
| Query2Box | 13.8 | 10.2 | 9.0 | 17.4 | 18.9 | 15.5 | 16.9 | 6.1 | 5.6 | **12.6** |
| BetaE | 2.3 | 1.5 | 1.1 | 2.8 | 2.0 | 1.4 | 1.4 | 1.9 | 1.0 | **1.7** |
| BetaE (+neg) | 2.5 | 1.8 | 1.2 | 3.7 | 2.7 | 1.6 | 2.2 | 2.5 | 1.3 | **2.2** |

### Negation structures, MRR (%)

| Model | 2in | 3in | inp | pin | pni | avg |
|---|---|---|---|---|---|---|
| BetaE (+neg) | 2.5 | 1.8 | 1.6 | 1.3 | 2.3 | **1.9** |

### Hits@k (%), average over the 9 positive structures

| Model | Hits@1 | Hits@3 | Hits@10 | train time (s) |
|---|---|---|---|---|
| GQE | 3.7 | 12.5 | 25.7 | 953 |
| Query2Box | 5.2 | 14.2 | 27.4 | 1853 |
| BetaE | 0.4 | 1.9 | 3.4 | 1650 |
| BetaE (+neg) | 0.8 | 2.2 | 4.3 | 1937 |

Nhận xét:

- Query2Box > GQE ở các dạng có giao (2i, 3i, pi) vì hộp thu nhỏ khi giao, đúng như thiết kế; cả hai yếu nhất ở các dạng có hợp (2u, up).
- **BetaE chưa train đủ** (MRR 1,7% với 6.000 bước): có thể do γ = 24 chưa hợp với thang KL (chưa thử giá trị khác) và số bước quá ít so với bản gốc (hàng trăm nghìn bước). Con số này không nói BetaE kém hơn hai mô hình kia.
- BetaE chạy được cả 5 dạng phủ định (MRR trung bình 1,9%), nhưng vì chưa train đủ nên chưa so sánh được với bản gốc.
- Chưa chạy nhiều seed, chưa tune siêu tham số.
