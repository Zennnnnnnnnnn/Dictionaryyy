from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Khởi tạo mô hình SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_sentence(sentence):
    # Chuyển câu thành vector embedding
    embedding = model.encode(sentence)
    return embedding

def calculate_similarity(sentence1, sentence2):
    # Encode hai câu thành các vector
    vec1 = encode_sentence(sentence1)
    vec2 = encode_sentence(sentence2)
    
    # Tính toán độ tương đồng cosine giữa hai vector
    similarity = cosine_similarity([vec1], [vec2])
    return similarity[0][0]

def find_high_similarity_pairs(list1, list2, threshold=0.5):
    # Trích xuất Description từ từng đối tượng trong danh sách `list1` và `list2`
    descriptions1 = [item['Description'] for item in list1]
    descriptions2 = [item['Description'] for item in list2]

    # Lấy embeddings cho các mô tả trong cả hai danh sách
    embeddings1 = [encode_sentence(desc) for desc in descriptions1]
    embeddings2 = [encode_sentence(desc) for desc in descriptions2]

    # Tính toán ma trận độ tương đồng giữa các embedding
    similarities = cosine_similarity(embeddings1, embeddings2)

    # Khởi tạo danh sách lưu trữ các cặp tương đồng cao
    pairs = []

    # Tạo hai tập hợp để theo dõi các chỉ số đã được ghép cặp
    matched1 = set()
    matched2 = set()

    # Duyệt qua từng hàng trong ma trận độ tương đồng
    for i, row in enumerate(similarities):
        # Duyệt qua từng cột trong hàng hiện tại
        for j, similarity in enumerate(row):
            # Kiểm tra nếu độ tương đồng lớn hơn ngưỡng và cả hai mô tả chưa được ghép cặp
            if similarity > threshold and i not in matched1 and j not in matched2:
                # Thêm các chỉ số vào các tập hợp đã được ghép cặp
                matched1.add(i)
                matched2.add(j)
                # Thêm cặp vào danh sách các cặp tương đồng cao
                pairs.append({
                    'Index_List1': i,
                    'Index_List2': j,
                    'Similarity': similarity
                })
                # In ra các cặp có độ tương đồng cao nhất
                #print(f"Added pair: Index_List1={i}, Index_List2={j}, Similarity={similarity:.4f}")

    return pairs
