# Tìm hiểu DAG trong KG

19/10/2026

## 1. Tổng quan

Tài liệu này tóm tắt cách DAG (Directed Acyclic Graph) được dùng trong Knowledge Graph (KG) để biểu diễn và trả lời các truy vấn logic nhiều bước, và chọn một mô hình nhỏ từ công trình lớn để chạy lại.

- **Task được giao:** tìm hiểu DAG trong KG, sau đó code một mô hình nhỏ của DAG; có thể lấy repo có sẵn nhưng phải từ công trình lớn và có dataset rõ ràng.
- **Hướng chọn:** truy vấn logic trên KG (complex logical query answering), nơi mỗi truy vấn là một DAG tính toán.
- **Repo gốc:** [snap-stanford/KGReasoning](https://github.com/snap-stanford/KGReasoning), code chính thức của GQE, Query2Box, BetaE.
- **Dataset:** FB15k-237 (14.505 entity, 474 relation).
- **Sản phẩm:** tài liệu này và repo [minhquan-maker/dag-in-kg](https://github.com/minhquan-maker/dag-in-kg).

## 2. DAG là gì

DAG là đồ thị có hướng, không có chu trình: đi theo chiều cạnh thì không bao giờ quay lại đỉnh đã đi qua. Một nhãn đỉnh có thể có nhiều cha, nên DAG tổng quát hóa cây.

### 2.1 Thuật toán cơ bản trên DAG

Bốn thao tác dưới đây được cài trong `dagkg.py` và chạy trên dữ liệu thật (82.115 đỉnh, 84.427 cạnh).

| Thao tác | Ý nghĩa | Cách làm |
| --- | --- | --- |
| Sắp xếp topo | Thứ tự mà cha luôn đứng trước con; phát hiện chu trình | Kahn: lấy đỉnh bậc vào 0, gỡ cạnh; nếu còn đỉnh thì có chu trình. Độ phức tạp O(V + E) |
| Bao đóng bắc cầu | Tập mọi tổ tiên của mỗi đỉnh | Quy hoạch động theo thứ tự topo: tổ tiên(c) = hợp của ({p} ∪ tổ tiên(p)) qua mọi cha p |
| Rút gọn bắc cầu | Giữ cạnh cơ bản, bỏ cạnh suy ra được | Bỏ (c, p) nếu p là tổ tiên của một cha khác của c |
| Tách train/val/test | Đo khả năng suy ra cạnh bắc cầu chưa thấy | Toàn bộ cạnh cơ bản vào train; trong phần còn lại của bao đóng: 10% test, 5% val |

### 2.2 Ví dụ

Trong WordNet, "Động vật ⊃ Thú ⊃ Chó" là một chuỗi. Cạnh Chó → Động vật là cạnh bắc cầu: nó không nằm trong tập cạnh cơ bản nhưng suy ra được từ hai cạnh kề nhau. 2.213 danh từ có hơn một cha, nên dữ liệu này thực sự là DAG chứ không phải cây.

## 3. DAG trong Knowledge Graph

DAG xuất hiện trong KG theo hai vai trò khác nhau; bài này đào sâu vai trò thứ nhất và điểm qua vai trò thứ hai.

| | Vai trò | DAG biểu diễn cái gì | Bài toán | Mô hình |
| --- | --- | --- | --- | --- |
| A | Phân cấp khái niệm (is-a) | Dữ liệu: đỉnh là khái niệm, cạnh là "là một loại của" | Link prediction trên cạnh bắc cầu | Poincaré, order embedding, box embedding |
| B | DAG tính toán của truy vấn logic | Truy vấn: đỉnh là tập thực thể, cạnh là phép toán | Complex logical query answering | GQE, Query2Box, BetaE |

### 3.1 Phân cấp khái niệm (vai trò A)

Quan hệ is-a là bắc cầu: Chó là Thú, Thú là Động vật thì Chó là Động vật. Một embedding tốt phải đọc ra được quan hệ này từ hình học, vì thế các mô hình khác nhau chọn hình học khác nhau:

- **Poincaré:** điểm trong quả cầu hyperbolic; khái niệm tổng quát gần gốc, khái niệm cụ thể gần biên.
- **Order embedding:** "u là con của v" khi u ≥ v ở mọi tọa độ. Bắc cầu có sẵn từ thứ tự từng phần.
- **Box embedding:** mỗi khái niệm là một hộp; con nằm trong hộp cha. Đúng ví dụ Động vật ⊃ Thú ⊃ Chó.

### 3.2 DAG tính toán của truy vấn logic (vai trò B)

Một truy vấn nhiều bước như "những đạo diễn có phim thuộc thể loại X **và** do diễn viên Y đóng" được viết thành DAG gồm ba phép toán trên tập thực thể:

- **Projection (p):** từ tập thực thể, đi theo một quan hệ để ra tập mới.
- **Intersection (i):** giao của các nhánh; đây là chỗ DAG khác chuỗi, vì một đỉnh nhận nhiều cạnh vào.
- **Union (u):** hợp của các nhánh.

Các dạng truy vấn chuẩn trên FB15k-237: 1p, 2p, 3p, 2i, 3i (huấn luyện) và thêm ip, pi, 2u, up (chỉ dùng khi đánh giá). Con số trong tên là số bước hoặc số nhánh. Mô hình nhúng truy vấn thành vector (GQE), hộp (Query2Box) hoặc phân phối Beta (BetaE) rồi duyệt DAG từ thực thể neo đến đỉnh đích.

## 4. Các công trình chính

Hai nhóm công trình tương ứng hai vai trò ở mục 3. Nhóm B là nền tảng cho truy vấn logic trên KG; nhóm A là nền tảng cho nhúng phân cấp.

| Nhóm | Công trình | Hội nghị | Ý chính |
| --- | --- | --- | --- |
| B | GQE (Hamilton et al.) | NeurIPS 2018 | Truy vấn là vector; projection và intersection là mạng nơ-ron |
| B | Query2Box (Ren et al.) | ICLR 2020 | Truy vấn là hộp; giao hộp tự nhiên hơn giao vector |
| B | BetaE (Ren & Leskovec) | NeurIPS 2020 | Truy vấn là phân phối Beta; hỗ trợ cả phủ định (NOT) |
| A | Poincaré Embeddings (Nickel & Kiela) | NeurIPS 2017 | Nhúng cây/DAG vào quả cầu hyperbolic |
| A | Hyperbolic Entailment Cones (Ganea et al.) | ICML 2018 | Nón hyperbolic cho quan hệ thứ tự; giao thức tách dữ liệu dùng trong repo |
| A | Order Embeddings (Vendrov et al.) | ICLR 2016 | Thứ tự từng phần trong orthant dương |
| A | Box Lattice Measures (Vilnis et al.) | ACL 2018 | Nhúng KG bằng hộp, "nằm trong" = suy ra |

Các con số tham chiếu trên FB15k-237 (MRR trung bình, %), theo README repo của [Trần Bình](https://github.com/btphuong3011/dag-kg-reasoning) trích từ paper: GQE 16,3; Query2Box 20,1; BetaE 20,9. Đây là mức chạy đầy đủ trên GPU; bản CPU thu nhỏ của bạn ấy đạt thấp hơn (GQE 6,6; Query2Box 5,8).

Paper ở nhóm B đã có repo chính thức [snap-stanford/KGReasoning](https://github.com/snap-stanford/KGReasoning). Repo của mình tập trung nhóm A để không trùng việc.

## 5. Mô hình và dataset chọn để code nhỏ

Repo [minhquan-maker/dag-in-kg](https://github.com/minhquan-maker/dag-in-kg) nhúng DAG is-a của WordNet danh từ bằng ba mô hình (Poincaré, order, box) và so với một baseline không học. Dataset và giao thức tách lấy từ công trình lớn (Nickel & Kiela 2017; Ganea et al. 2018) nên so sánh được với paper.

**Vì sao chọn hướng này:** dataset rõ ràng, chạy được trên CPU, và DAG hiện ra trực tiếp trong dữ liệu. Dùng đúng ví dụ Thú ⊃ Chó của phần lý thuyết.

### 5.1 Dataset: WordNet 3.0 danh từ

Code tự dựng lại từ file `data.noun` gốc (`tools/build_wordnet_dag.py`) và khớp paper ở những con số paper công bố.

| Chỉ số | Repo này | Paper |
| --- | --- | --- |
| Synset danh từ | 82.115 | 82.115 |
| Cạnh is-a (hypernym + instance hypernym) | 84.427 | |
| Nút có hơn 1 cha (không phải cây) | 2.213 | |
| Cạnh bao đóng bắc cầu | 743.241 | 743.241 |

Sau khi tách: ở mức train thêm 50% cạnh bắc cầu: train 364.388, val 32.943, test 65.887 cạnh.

### 5.2 Bốn mô hình (dim = 10)

| Mô hình | Đọc quan hệ "u là con của v" từ | Huấn luyện |
| --- | --- | --- |
| popularity | Số lần v làm cha trong train (baseline) | Không |
| poincare | Khoảng cách hyperbolic, v gần gốc hơn u | Riemannian SGD |
| order | u ≥ v ở mọi tọa độ; điểm phạt = mức vi phạm | Margin loss |
| box | Hộp u nằm trong hộp v; điểm phạt = phần hộp tràn ra ngoài | Margin loss |

## 6. Cách chạy và kết quả

Box cho kết quả tốt nhất ở cả hai mức dữ liệu (hard F1 0,81 khi train chỉ có cạnh cơ bản, 0,95 khi thêm 50% cạnh bắc cầu). Kết quả dùng 1 seed, 30 epoch.

### 6.1 Cách chạy

```bash
pip install -r requirements.txt
bash run_all.sh      # khoảng 15 phút trên CPU 4 nhân, không cần GPU
python3 train_eval.py --model box --hard_neg --frac 0.5   # chạy lẻ một mô hình
```

### 6.2 Đánh giá

- **Hai chế độ negative:** *easy* chọn nút thay thế ngẫu nhiên đều; *hard* chọn theo phân phối các nút từng làm cha trong train, nên "nút này hay làm cha" không còn là manh mối.
- **Hai độ đo:** F1 phân loại cạnh thật với cạnh giả (50/50, ngưỡng chọn trên val); MRR của cha thật so với 100 ứng viên.

### 6.3 Kết quả trên tập test

| % cạnh bắc cầu có trong train | Mô hình | easy F1 | easy MRR | hard F1 | hard MRR |
| --- | --- | --- | --- | --- | --- |
| 0% | popularity | 0,797 | 0,524 | 0,680 | 0,039 |
| 0% | poincare | 0,698 | 0,205 | 0,693 | 0,194 |
| 0% | order | 0,817 | 0,864 | 0,798 | 0,540 |
| 0% | box | 0,822 | 0,965 | 0,808 | 0,861 |
| 50% | popularity | 0,808 | 0,927 | 0,679 | 0,156 |
| 50% | poincare | 0,945 | 0,726 | 0,888 | 0,480 |
| 50% | order | 0,887 | 0,993 | 0,819 | 0,852 |
| 50% | box | 0,948 | 0,999 | 0,953 | 0,944 |

### 6.4 Nhận xét

- **Đọc cột hard.** Ở easy, baseline popularity đã đạt F1 ≈ 0,80 và MRR 0,93 (50%), nhưng ở hard chỉ còn F1 ≈ 0,68 và MRR 0,16.
- **Poincaré cần thấy cạnh bắc cầu.** Khi train chỉ có cạnh cơ bản, Poincaré chưa hơn baseline (hard F1 0,693 so với 0,680).
- **Poincaré thấp hơn số trong paper** và thấp hơn Box ở đây. Mình chỉ train 30 epoch, dim 10, 1 seed, chọn lr = 30 bằng thử trên val và dùng negative trộn "hard" cho cả 3 mô hình; paper train hàng trăm epoch với setup khác. Bảng này **không** dùng để kết luận hyperbolic kém hơn box; nó chỉ kiểm tra pipeline chạy đúng và mô hình học được cấu trúc DAG thật.
- **Chênh lệch vài điểm %** với 1 seed chưa đủ để kết luận.

## 7. Kết luận và hướng tiếp theo

DAG trong KG vừa là cấu trúc dữ liệu (phân cấp is-a) vừa là cấu trúc tính toán (truy vấn logic); repo này chứng minh vai trò thứ nhất trên dữ liệu thật, với box embedding phù hợp nhất trong các mô hình đã thử.

**Hướng tiếp theo**

- [ ] Chạy nhiều seed và nhiều epoch hơn để kết luận được Poincaré so với Box.
- [ ] Thử dim lớn hơn 10 và xem kết quả thay đổi thế nào.
- [ ] Nối hai nhóm: dùng embedding phân cấp làm nguồn thông tin cho truy vấn logic trên KG.
- [ ] Đối chiếu với kết quả GQE / Query2Box trên FB15k-237 của Trần Bình.

**Lưu ý về nguồn:** không truy cập được Google Doc mẫu từ môi trường chạy, nên bố cục doc này dựa trên cấu trúc báo cáo của repo mẫu; có thể cần chỉnh lại cho giống hoàn toàn.

### Nguồn

- [dag-kg-reasoning (Trần Bình)](https://github.com/btphuong3011/dag-kg-reasoning), đã mở ngày 2026-10-19: số liệu GQE / Query2Box / BetaE trên FB15k-237.
- [snap-stanford/KGReasoning](https://github.com/snap-stanford/KGReasoning): repo chính thức của nhóm B (chưa mở được từ môi trường này).
- [minhquan-maker/dag-in-kg](https://github.com/minhquan-maker/dag-in-kg): code và kết quả ở mục 5 và 6.
- Nickel & Kiela, Poincaré Embeddings, NeurIPS 2017; Ganea et al., Hyperbolic Entailment Cones, ICML 2018; Vendrov et al., Order-Embeddings, ICLR 2016; Vilnis et al., Box Lattice Measures, ACL 2018.
