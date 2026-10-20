# dag-in-kg: DAG như phân cấp khái niệm (is-a) trong Knowledge Graph

Bài tìm hiểu DAG trong KG, phần **§2.1 – phân cấp khái niệm**: embedding một DAG is-a sao cho "u là một loại của v" được đọc ra từ hình học. Phần §2.2 (DAG tính toán của truy vấn logic: GQE/Query2Box/BetaE) nằm trong thư mục [`kgqa/`](kgqa/README.md), chạy trên FB15k-237; thư mục gốc làm hướng còn lại.

**Dữ liệu**: WordNet 3.0 danh từ, dữ liệu chuẩn của Poincaré Embeddings (Nickel & Kiela, NeurIPS 2017) và Hyperbolic Entailment Cones (Ganea et al., ICML 2018). Code tự dựng lại từ file `data.noun` gốc và kiểm tra khớp paper:

| | Repo này | Paper |
|---|---|---|
| Synset danh từ | 82.115 | 82.115 |
| Cạnh is-a (hypernym + instance hypernym) | 84.427 | |
| Nút có > 1 cha (tức không phải cây) | 2.213 | |
| Cạnh bao đóng bắc cầu | 743.241 | 743.241 |

## Pipeline

1. `tools/build_wordnet_dag.py`: tải WordNet, parse cạnh (con → cha).
2. `dagkg.py`: các thuật toán ở mục 1.3 của tài liệu, áp lên dữ liệu thật: sắp xếp topo Kahn (có kiểm tra không chu trình), bao đóng bắc cầu bằng quy hoạch động theo thứ tự topo, rút gọn bắc cầu.
3. Tách dữ liệu theo Ganea et al.: toàn bộ cạnh cơ bản (rút gọn bắc cầu) luôn nằm trong train; trong các cạnh bao đóng còn lại, 10% test, 5% val, phần train lấy thêm 0% hoặc 50% của số còn lại. Test thể hiện đúng khả năng **suy ra cạnh bắc cầu chưa thấy**.
4. `models.py`: ba cách nhúng DAG, dim = 10:
   - `poincare`: điểm trong quả cầu Poincaré, Riemannian SGD (Nickel & Kiela 2017).
   - `order`: order embedding, "u là con của v" khi u ≥ v từng tọa độ (Vendrov et al. 2016).
   - `box`: mỗi khái niệm là một hộp, con nằm trong hộp cha (đúng ví dụ Động vật ⊃ Thú ⊃ Chó trong tài liệu).
   - `popularity`: baseline không học gì, chỉ đếm số lần v làm cha trong train.
5. `train_eval.py`: huấn luyện + đánh giá link prediction.

## Chạy

```bash
pip install -r requirements.txt
bash run_all.sh      # ~15 phút trên CPU 4 nhân, không cần GPU; in bảng kết quả cuối cùng
```

Chạy lẻ: `python3 train_eval.py --model box --hard_neg --frac 0.5`.

## Đánh giá

Hai chế độ negative (cặp không phải tổ tiên-hậu duệ), vì lúc đầu mình thấy negative ngẫu nhiên quá dễ:
- **easy**: nút thay thế chọn đều ngẫu nhiên.
- **hard**: nút thay thế chọn theo phân phối các nút từng làm cha trong train, nên "nút này hay làm cha" không còn là manh mối. Baseline popularity phải rơi về mức ngẫu nhiên.

Hai độ đo: **F1** phân loại cạnh thật vs cạnh giả (50/50, bóp méo con hoặc cha, ngưỡng chọn trên val); **MRR** của cha thật so với 100 ứng viên.

## Kết quả (tập test, 1 seed, 30 epoch, dim 10)

| % cạnh bắc cầu có trong train | Mô hình | easy F1 | easy MRR | hard F1 | hard MRR |
|---|---|---|---|---|---|
| 0% | popularity | 0,797 | 0,524 | 0,680 | 0,039 |
| 0% | poincare | 0,698 | 0,205 | 0,693 | 0,194 |
| 0% | order | 0,817 | 0,864 | 0,798 | 0,540 |
| 0% | box | 0,822 | 0,965 | 0,808 | 0,861 |
| 50% | popularity | 0,808 | 0,927 | 0,679 | 0,156 |
| 50% | poincare | 0,945 | 0,726 | 0,888 | 0,480 |
| 50% | order | 0,887 | 0,993 | 0,819 | 0,852 |
| 50% | box | 0,948 | 0,999 | 0,953 | 0,944 |

Nhận xét:
- Chế độ hard cần thiết: ở easy, baseline popularity đã đạt F1 ≈ 0,80 và MRR 0,93, trong khi hard cho nó F1 ≈ 0,68 và MRR 0,16. Vì vậy hãy đọc cột hard.
- Box cho kết quả tốt nhất ở cả hai mức dữ liệu (hard F1 0,81 / 0,95).
- Khi train chỉ có cạnh cơ bản (0%), Poincaré chưa hơn baseline (hard F1 0,693 vs 0,680). Mô hình cần thấy cạnh bắc cầu thì hình học mới có tác dụng.
- Poincaré **thấp hơn** số trong paper (và thấp hơn Box ở đây). Mình chỉ train 30 epoch, dim 10, 1 seed, chọn lr (30) bằng thử trên val và dùng negative trộn "hard" cho cả 3 mô hình. Paper train hàng trăm epoch với setup khác, nên bảng này **không** dùng để kết luận hyperbolic kém hơn box. Nó chỉ kiểm tra pipeline chạy đúng và các mô hình học được cấu trúc DAG thật.
- Một seed nên chênh lệch nhỏ (vài điểm %) chưa đủ để kết luận.

## Tài liệu tham khảo

- Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical Representations*, NeurIPS 2017.
- Ganea, Bécigneul & Hofmann, *Hyperbolic Entailment Cones for Learning Hierarchical Embeddings*, ICML 2018.
- Vendrov et al., *Order-Embeddings of Images and Language*, ICLR 2016.
- Vilnis et al., *Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures*, ACL 2018.
