🔎Xây dựng hệ thống dự đoán nấm ăn được hay nấm độc dựa trên bộ dữ liệu Mushroom.
Quy trình gồm:

Phân tích dữ liệu: dùng Python, pandas, seaborn để thống kê, vẽ biểu đồ phân phối các thuộc tính nấm 

Huấn luyện mô hình: tự cài đặt thuật toán ID3 (Decision Tree) để huấn luyện trên bộ dữ liệu nấm, đánh giá bằng confusion matrix, classification report, accuracy rồi lưu mô hình dưới dạng file mushroom_model.pkl 


Xây dựng ứng dụng dự đoán: tạo giao diện để người dùng nhập vào thuộc tính nấm (odor, cap-color, gill-color), sau đó mô hình dự đoán ra kết quả:

e → Edible (ăn được).

p → Poisonous (nấm độc).

hoặc Không xác định nếu dữ liệu không khớp.

Ứng dụng có 2 phiên bản:

Giao diện desktop: viết bằng tkinter (Python GUI) 

Giao diện web: viết bằng Flask với HTML template 

⚙️ Công nghệ và ngôn ngữ sử dụng

Ngôn ngữ chính: Python.

Thư viện phân tích: pandas, numpy, matplotlib, seaborn.

Thư viện machine learning: scikit-learn (chỉ hỗ trợ chia tập train/test và đánh giá; cây quyết định ID3 được cài đặt thủ công).

Thư viện giao diện:

tkinter cho ứng dụng desktop.

Flask + HTML/CSS cho ứng dụng web.

Công cụ hỗ trợ: pickle để lưu/trích xuất mô hình đã huấn luyện.

📊 Thuật toán sử dụng

ID3 Decision Tree:

Tính entropy và information gain cho từng thuộc tính.

Chọn thuộc tính có information gain lớn nhất để chia nhánh.

<img width="954" height="474" alt="image" src="https://github.com/user-attachments/assets/f82a4f47-eba0-4bf8-bfe1-485247401de0" />

<img width="935" height="503" alt="image" src="https://github.com/user-attachments/assets/5cf7d475-d4d0-491e-afc9-3f6afc49898a" />

Đệ quy xây dựng cây cho đến khi các lá là nhãn e (edible) hoặc p (poisonous).

Dự đoán bằng cách duyệt cây quyết định theo các thuộc tính đầu vào.
