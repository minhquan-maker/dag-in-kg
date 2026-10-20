---
title: "TÌM HIỂU DAG TRONG KNOWLEDGE GRAPH"
---

# 1. DAG là gì, dùng để làm gì

**Định nghĩa.** DAG (Directed Acyclic Graph) là đồ thị có hướng mà không có chu trình: xuất phát từ một đỉnh bất kỳ, cứ đi theo chiều mũi tên thì không bao giờ quay về được đỉnh đó. Khác cây ở chỗ một đỉnh được phép có nhiều cha.

**Mấy tính chất mình cần nhớ:**

- *Thứ tự topo.* Đồ thị có hướng là DAG khi và chỉ khi xếp được các đỉnh thành một dãy sao cho mọi cạnh u → v đều có u đứng trước v. Có dãy này thì xử lý đỉnh nào cũng đã có sẵn kết quả của các đỉnh nó phụ thuộc.
- *Nguồn và đích.* DAG hữu hạn luôn có ít nhất một đỉnh không có cạnh vào (nguồn) và một đỉnh không có cạnh ra (đích).
- *Thứ tự bộ phận.* "Từ u đi tới được v" là quan hệ bắc cầu và không có hai đỉnh nào đi tới nhau, tức là một thứ tự bộ phận chặt. Từ đây có hai phép hay dùng: **bao đóng bắc cầu** (thêm cạnh u → w nếu có đường u → … → w) và **rút gọn bắc cầu** (bỏ những cạnh suy ra được, chỉ giữ phần khung tối thiểu).

**Thuật toán cơ bản:**

- *Sắp xếp topo (Kahn).* Mỗi bước lấy một đỉnh có bậc vào 0, rồi xóa các cạnh đi ra của nó. Độ phức tạp O(V + E). Nếu còn đỉnh mà không đỉnh nào có bậc vào 0 thì đồ thị có chu trình. Cách khác là DFS, ghi đỉnh lúc duyệt xong rồi đảo ngược dãy.
- *Quy hoạch động trên DAG.* Duyệt theo thứ tự topo, giá trị mỗi đỉnh tính từ các đỉnh đứng trước. Dùng được cho đường đi dài nhất, đếm số đường đi, lan truyền giá trị từ nguồn tới đích. Bao đóng bắc cầu cũng làm kiểu này: tổ tiên(c) = hợp của {p} ∪ tổ tiên(p) qua mọi cha p của c.
- *Tổ tiên chung gần nhất (LCA).* Trong cây thì LCA duy nhất, còn trong DAG hai đỉnh có thể có nhiều LCA. Hay dùng để đo hai khái niệm gần nhau cỡ nào trong một phân cấp.

# 2. DAG nằm ở đâu trong KG

Bản thân một KG thường có chu trình (A quen B, B quen A, ...) nên không phải DAG. DAG chỉ hiện ra ở một số phần và một số cách dùng KG. Mình gom được ba chỗ.

## 2.1. Phân cấp khái niệm (is-a)

Các khái niệm trong KG xếp theo quan hệ "là một loại của" (is-a; trong RDF/OWL là `subClassOf`). Chó là một loại Thú, Thú là một loại Động vật. Vẽ mũi tên từ con lên cha thì ra DAG:

- Không có chu trình, vì không khái niệm nào là một loại của chính nó.
- Không phải cây, vì một khái niệm có nhiều cha (Chó vừa là Thú vừa là Vật nuôi).
- Suy diễn bắc cầu: Chó → Thú → Động vật nên Chó → Động vật. Đi ngược lên theo mũi tên là ra mọi khái niệm rộng hơn.

Từ đây có bài toán **nhúng phân cấp**: biểu diễn mỗi khái niệm bằng một hình trong không gian sao cho "cha chứa con" đọc ra được từ hình học.

- *Order embedding:* u là con của v khi u ≥ v ở mọi tọa độ. Bắc cầu có sẵn vì ≥ từng tọa độ là bắc cầu.
- *Box embedding:* mỗi khái niệm là một hộp, hộp Động vật chứa hộp Thú, hộp Thú chứa hộp Chó.
- *Poincaré embedding:* dùng không gian hyperbolic, càng ra xa tâm càng "rộng" nên xếp được rất nhiều nhánh con. Khái niệm tổng quát nằm gần tâm, khái niệm cụ thể nằm gần biên.
- *Hyperbolic entailment cones:* mỗi khái niệm có một hình nón, con nằm trong nón của cha.

## 2.2. Truy vấn logic nhiều bước

Câu hỏi phức tạp trên KG viết được thành công thức logic, và cấu trúc tính toán của công thức đó là một DAG: thực thể đã biết là đỉnh nguồn, đáp án là đỉnh đích, ở giữa là biến trung gian hoặc phép toán logic. Phần 3 và 4 đi sâu vào cái này.

## 2.3. Kế hoạch suy luận trong KG-RAG

KG-RAG hay tách câu hỏi khó thành các câu nhỏ. Ví dụ "Einstein và Marie Curie, ai sinh trước?" tách thành (1) Einstein sinh năm nào, (2) Marie Curie sinh năm nào, (3) so sánh hai năm. Câu (3) phải chờ (1) và (2); (1) với (2) chạy song song được. Các câu nhỏ cùng quan hệ "phải chờ" tạo thành một DAG, hệ thống trả lời theo thứ tự topo.

# 3. Trả lời truy vấn logic phức tạp trên KG

## 3.1. Câu hỏi thành công thức

Ví dụ: "Những ai đoạt giải Nobel Vật lý và sinh ra ở một thành phố của Đức?"

q = V? . ∃V : thành_phố_của(V, Đức) ∧ sinh_tại(V?, V) ∧ đạt_giải(V?, Nobel Vật lý)

Đọc là: tìm V? sao cho có một thành phố V của Đức mà V? sinh tại V, và V? đạt giải Nobel Vật lý. V? là đáp án, V là biến trung gian (không cần biết cụ thể là thành phố nào).

Các phép dùng để dựng truy vấn:

- **Chiếu (projection):** đi theo một quan hệ từ tập thực thể này sang tập khác. Ví dụ từ {Đức} đi theo quan hệ ngược thành_phố_của⁻¹ ra tập các thành phố của Đức.
- **Giao (∧):** giữ thực thể thỏa mọi nhánh.
- **Hợp (∨):** thỏa ít nhất một nhánh.
- **Phủ định (¬):** lấy phần bù.

Truy vấn chỉ dùng ∃, ∧, ∨ gọi là EPFO (existential positive first-order). Thêm ¬ thì rộng hơn.

## 3.2. Từ câu hỏi sang DAG

Mỗi câu hỏi vẽ thành một DAG: điểm xuất phát là các thực thể đã biết (Đức, Giải Nobel Vật lý), mỗi mũi tên là một bước đi theo quan hệ, chỗ hai nhánh gặp nhau là phép giao, điểm cuối V? là đáp án. Trả lời = đi qua DAG theo thứ tự topo, xong bước trước mới tới bước sau. Tên các dạng câu hỏi đọc theo hình dạng DAG: 1p, 2p, 3p là chuỗi 1, 2, 3 phép chiếu; 2i, 3i là giao của 2, 3 nhánh; ip, pi, 2u, up là các dạng ghép (giao rồi chiếu, chiếu rồi giao, hợp của hai nhánh, hợp rồi chiếu).

## 3.3. Vì sao không duyệt đồ thị trực tiếp

- **KG thiếu cạnh.** FB15k-237 hay Wikidata thiếu rất nhiều sự thật đúng. Nếu KG thiếu cạnh "Einstein sinh tại Ulm" thì đi trực tiếp sẽ không bao giờ ra Einstein.
- **Tập trung gian nổ rất nhanh.** Một nước có hàng nghìn thành phố, mỗi thành phố có rất nhiều người sinh ra, nên câu nhiều bước tốn thời gian.

Cách giải là **nhúng truy vấn (query embedding)**: đưa thực thể và câu hỏi vào cùng một không gian vector. Mỗi phép chiếu, giao, hợp là một phép tính trên vector và được học từ dữ liệu. Đi hết DAG thì có vector của câu hỏi, đáp án là các thực thể gần nó nhất. Vì học quy luật chung của KG nên mô hình đoán được cả cạnh bị thiếu, và mỗi bước chỉ là một phép tính vector nên chi phí không phụ thuộc số thực thể trung gian.

# 4. GQE, Query2Box, BetaE

Cả ba làm cùng một khung: mỗi thực thể và quan hệ có embedding học được; đi theo DAG của câu hỏi, dùng phép chiếu và phép giao để biến embedding thực thể xuất phát thành embedding câu hỏi; tính khoảng cách từ câu hỏi tới mọi thực thể, càng gần càng có khả năng là đáp án. Khác nhau ở **hình dạng của embedding câu hỏi**.

| | GQE | Query2Box | BetaE |
| --- | --- | --- | --- |
| Câu hỏi là | một điểm | một hộp (tâm + offset) | một bộ d phân phối Beta |
| Chiếu | q + r | dời tâm, nới rộng hộp | MLP ra (α, β) mới |
| Giao | trung bình có trọng số (attention) | tâm: trung bình có trọng số; kích thước: lấy nhỏ nhất rồi nhân hệ số trong (0, 1) | nhân các đường cong có trọng số |
| Hợp | không làm trực tiếp | đưa về DNF, lấy khoảng cách nhỏ nhất tới các hộp | DNF hoặc De Morgan |
| Phủ định | không | không | (α, β) → (1/α, 1/β) |
| Khoảng cách | γ − ‖e − q‖₁ | dist_out + α·dist_in | KL divergence |
| Công bố | NeurIPS 2018 | ICLR 2020 | NeurIPS 2020 |

## 4.1. GQE (Hamilton và cộng sự)

Truy vấn là một vector. Chiếu là cộng vector quan hệ: Đức = (1, 2), quan hệ = (3, 0) thì sau chiếu được (4, 2). Giao là trung bình có trọng số, trọng số do một mạng attention nhỏ tự học (nhánh đáng tin hơn thì trọng số lớn hơn). Điểm của thực thể e là γ − ‖e − q‖₁, với ‖·‖₁ là khoảng cách L1 (cộng trị tuyệt đối chênh lệch từng tọa độ), γ là hằng số.

Hạn chế: câu hỏi thường có nhiều đáp án mà một điểm thì khó bao được cả tập; GQE cũng không làm trực tiếp được phép hợp.

## 4.2. Query2Box (Ren, Hu & Leskovec)

Truy vấn là một hộp trong không gian d chiều, gồm tâm và offset (nửa độ dài cạnh, không âm). Ví dụ tâm (4, 2), offset (1; 0,5) là hộp có x từ 3 đến 5 và y từ 1,5 đến 2,5. Thực thể nằm trong hộp là đáp án, hợp với trường hợp đáp án là cả một tập.

- Chiếu: dời tâm theo vector quan hệ và nới hộp ra, vì đi qua một quan hệ thì số đáp án thường tăng.
- Giao: hộp mới luôn nhỏ hơn mọi hộp ban đầu, giống giao tập hợp.
- Khoảng cách: dist_out + α·dist_in. Phần nằm ngoài hộp phạt nặng, bên trong phạt nhẹ (α = 0,02 trong cấu hình gốc).
- Hợp: hợp của các hộp không còn là hộp, nên viết lại câu hỏi để "hoặc" nằm ở bước cuối (dạng chuẩn tắc tuyển, DNF). Ví dụ "(A hoặc B) rồi đi theo r" thành "(A đi theo r) hoặc (B đi theo r)"; mỗi vế là một hộp, thực thể chỉ cần gần một trong các hộp.

## 4.3. BetaE (Ren & Leskovec)

Phân phối Beta là một đường cong trên [0, 1] có hình dạng do hai số α, β quyết định: Beta(2, 5) lệch trái, Beta(5, 2) lệch phải, Beta(1, 1) phẳng. Mỗi thực thể và mỗi câu hỏi là d đường cong như vậy, mỗi chiều một đường.

- Giao: nhân các đường cong với trọng số, chỉ giữ vùng mà mọi nhánh đều cao. Kết quả vẫn là Beta, với α, β là tổng có trọng số của α, β ban đầu.
- Phủ định: (α, β) → (1/α, 1/β) biến vùng mật độ cao thành thấp và ngược lại; áp dụng hai lần thì về như cũ. Đây là lý do BetaE là mô hình duy nhất trong ba cái làm được câu hỏi có phủ định.
- Hợp: DNF hoặc De Morgan, A ∨ B = ¬(¬A ∧ ¬B).
- Đường cong càng trải rộng thì câu hỏi càng nhiều đáp án, nên BetaE ước lượng được câu hỏi rộng hay hẹp.

## 4.4. Huấn luyện và đánh giá

- **Huấn luyện:** lấy truy vấn q, một đáp án đúng v và k thực thể ngẫu nhiên không phải đáp án v′₁…v′ₖ (mẫu âm). Hàm mất mát L = −log σ(γ − d(v, q)) − (1/k)·Σᵢ log σ(d(v′ᵢ, q) − γ), d là khoảng cách, σ là sigmoid. Vế đầu nhỏ khi đáp án đúng gần câu hỏi, vế sau nhỏ khi đáp án sai xa. Tức là kéo đáp án đúng lại gần, đẩy đáp án sai ra xa.
- **Easy và hard answers:** đồ thị chia train ⊂ valid ⊂ test. Với truy vấn test, đáp án tìm được bằng cách duyệt đồ thị valid là "easy"; đáp án chỉ xuất hiện khi thêm cạnh test là "hard", tức mô hình phải tự suy ra cạnh thiếu. Chỉ hard answers được tính điểm, easy bị loại khỏi bảng xếp hạng (filtered setting).
- **Độ đo:** với mỗi hard answer lấy thứ hạng của nó trong danh sách mọi thực thể. MRR là trung bình 1/thứ hạng, Hits@k là tỉ lệ đáp án nằm trong top-k. Báo cáo riêng từng dạng truy vấn.

Phần 5 là mình tự chạy lại ba mô hình này trên FB15k-237; phần 6 chạy thêm hướng phân cấp is-a ở mục 2.1.

# 5. Tự chạy: GQE, Query2Box, BetaE trên FB15k-237

Mình không fork repo của Trần Bình mà viết lại từ đầu (thư mục `kgqa/` trong repo), giữ cùng ý tưởng: sinh bộ câu hỏi kiểu BetaE từ FB15k-237 rồi train và đo MRR trên hard answers.

## 5.1. Mình làm khác chỗ nào

- Mỗi dạng câu hỏi là một DAG nhỏ khai báo trong `dagq.py` (anchor, chiếu, giao, phủ định), một bộ thực thi duyệt DAG theo thứ tự topo rồi gọi phép của mô hình. Ba mô hình dùng chung bộ này.
- Sinh câu hỏi bằng cách chọn một thực thể đáp án rồi đi ngược qua DAG; với valid/test ép một cạnh bị giấu (held-out) vào đường đi để có hard answers.
- 40.000 câu train mỗi dạng, 1.000 câu test mỗi dạng; số cạnh train/valid/test khớp BetaE (272.115 / 17.526 / 20.438).
- Thêm 5 dạng có phủ định (2in, 3in, inp, pin, pni) cho BetaE.

## 5.2. Kết quả

CPU 4 nhân, 1 seed: GQE 12.000 bước, Query2Box 8.000, BetaE 6.000, batch 128, dim 256. MRR (%) trên hard answers của tập test, trung bình 9 dạng (avg):

| Mô hình | 1p | 2p | 3p | 2i | 3i | ip | pi | 2u | up | avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQE | 11,5 | 10,2 | 9,1 | 15,1 | 14,9 | 14,9 | 14,5 | 5,3 | 4,8 | **11,1** |
| Query2Box | 13,8 | 10,2 | 9,0 | 17,4 | 18,9 | 15,5 | 16,9 | 6,1 | 5,6 | **12,6** |
| BetaE | 2,3 | 1,5 | 1,1 | 2,8 | 2,0 | 1,4 | 1,4 | 1,9 | 1,0 | **1,7** |

BetaE train thêm cả dạng phủ định đạt 1,9% trung bình trên 5 dạng 2in, 3in, inp, pin, pni.

- Query2Box hơn GQE ở các dạng có giao (2i, 3i, pi), đúng ý của hộp: giao hộp luôn nhỏ lại.
- Cả hai yếu nhất ở dạng có hợp (2u, up).
- BetaE mới train 6.000 bước nên thấp. Có thể do γ = 24 chưa hợp với thang KL, mình chưa thử giá trị khác, nên không kết luận được BetaE kém hơn.
- Một seed, chưa tune siêu tham số.

# 6. Chạy thêm: nhúng DAG is-a của WordNet

Repo: github.com/minhquan-maker/dag-in-kg. Mục tiêu: kiểm tra xem mấy cách nhúng ở 2.1 có thật sự đọc ra được "u là một loại của v" từ hình học hay không, trên một DAG thật chứ không phải ví dụ đồ chơi.

## 6.1. Dữ liệu

WordNet 3.0 danh từ, dùng đúng bộ của Poincaré Embeddings (Nickel & Kiela, NeurIPS 2017) và Hyperbolic Entailment Cones (Ganea và cộng sự, ICML 2018). Mình tự dựng lại từ file `data.noun` gốc rồi so với số trong paper:

| | Repo này | Paper |
| --- | --- | --- |
| Synset danh từ | 82.115 | 82.115 |
| Cạnh is-a (hypernym + instance hypernym) | 84.427 | |
| Nút có hơn 1 cha | 2.213 | |
| Cạnh bao đóng bắc cầu | 743.241 | 743.241 |

Hai số khớp paper thì yên tâm là dựng đúng. 2.213 nút có nhiều cha, nghĩa là dữ liệu này đúng là DAG chứ không phải cây.

## 6.2. Cách làm

Các thuật toán ở mục 1 được cài trong `dagkg.py` và chạy trên dữ liệu thật: Kahn có kiểm tra chu trình, bao đóng bằng quy hoạch động theo thứ tự topo, rút gọn bắc cầu.

Cách chia dữ liệu theo Ganea: toàn bộ cạnh cơ bản (sau rút gọn bắc cầu) luôn nằm trong train. Trong các cạnh bao đóng còn lại, 10% làm test, 5% làm val, và phần train lấy thêm 0% hoặc 50% số còn lại. Nhờ vậy tập test đo đúng khả năng **suy ra cạnh bắc cầu chưa thấy**.

Bốn mô hình, đều dim = 10:

| Mô hình | Đọc "u là con của v" từ | Huấn luyện |
| --- | --- | --- |
| popularity | số lần v làm cha trong train (baseline, không học) | không |
| poincare | khoảng cách hyperbolic, v gần gốc hơn u | Riemannian SGD |
| order | u ≥ v từng tọa độ, điểm phạt là mức vi phạm | margin loss |
| box | hộp u nằm trong hộp v, điểm phạt là phần hộp tràn ra ngoài | margin loss |

Đánh giá có hai kiểu negative. *Easy*: nút thay thế chọn ngẫu nhiên đều. *Hard*: chọn theo phân phối các nút từng làm cha trong train, để "nút này hay làm cha" không còn là manh mối. Lý do có kiểu hard: lúc đầu mình thấy negative ngẫu nhiên quá dễ, baseline popularity không học gì mà vẫn được F1 khoảng 0,80. Hai độ đo: F1 phân loại cạnh thật với cạnh giả (50/50, ngưỡng chọn trên val) và MRR của cha thật so với 100 ứng viên.

Chạy: `bash run_all.sh`, khoảng 15 phút trên CPU 4 nhân, không cần GPU.

## 6.3. Kết quả

Tập test, 1 seed, 30 epoch, dim 10.

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

Mấy điều rút ra:

- Phải đọc cột hard. Ở easy, popularity đã được F1 khoảng 0,80 và MRR 0,93 (mức 50%), sang hard còn F1 khoảng 0,68 và MRR 0,16.
- Box tốt nhất ở cả hai mức dữ liệu (hard F1 0,81 và 0,95). Hộp nằm trong hộp khớp đúng ví dụ Động vật ⊃ Thú ⊃ Chó.
- Khi train chỉ có cạnh cơ bản (0%), Poincaré chưa hơn baseline (hard F1 0,693 so với 0,680). Mô hình cần thấy cạnh bắc cầu thì hình học mới có tác dụng.
- Poincaré **thấp hơn** số trong paper và thấp hơn Box ở đây. Mình chỉ train 30 epoch, dim 10, 1 seed, chọn lr = 30 bằng cách thử trên val và dùng negative trộn kiểu hard cho cả ba mô hình; paper train hàng trăm epoch với setup khác. Nên bảng này **không** dùng để kết luận hyperbolic kém hơn box, nó chỉ cho thấy pipeline chạy đúng và mô hình học được cấu trúc DAG thật.
- Chỉ một seed nên chênh vài điểm % chưa đủ để kết luận.

# 7. Rút ra

DAG xuất hiện trong KG ở hai chỗ chính: phân cấp khái niệm (is-a) và sơ đồ tính toán của câu hỏi nhiều bước (kèm kế hoạch suy luận trong KG-RAG). Với truy vấn logic, GQE, Query2Box, BetaE biến từng phép trên DAG thành phép tính học được nên trả lời được cả những câu cần cạnh bị thiếu; cách biểu diễn câu hỏi đi từ điểm, sang hộp, sang phân phối Beta, và mỗi bước làm được thêm một thứ (đáp án là cả tập, rồi phủ định, rồi độ rộng hẹp của câu hỏi). Chạy lại trên FB15k-237 thì Query2Box nhỉnh hơn GQE (MRR trung bình 12,6 so với 11,1), còn BetaE cần train lâu hơn mới đánh giá được. Với phân cấp, thử trên WordNet thì hộp là cách nhúng ăn nhất trong ba cách đã chạy.

Còn muốn làm tiếp: chạy nhiều seed và nhiều epoch để biết chắc Poincaré với Box ai hơn; thử dim lớn hơn 10; thử dùng embedding phân cấp làm thông tin thêm cho truy vấn logic.
