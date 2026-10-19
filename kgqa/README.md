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
