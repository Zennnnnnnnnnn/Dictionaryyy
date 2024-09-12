import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Hàm để lấy embeddings từ mô hình GTE-Large
def get_embeddings(texts):
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
    model = AutoModel.from_pretrained("thenlper/gte-large")

    # Tokenize the input texts
    batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**batch_dict)
    
    # Average pooling
    last_hidden_states = outputs.last_hidden_state
    attention_mask = batch_dict['attention_mask']
    embeddings = average_pool(last_hidden_states, attention_mask)
    
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.numpy()

# Hàm để thực hiện average pooling trên các embeddings
def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Hàm để tìm cặp mô tả có độ tương đồng cao
def find_high_similarity_pairs(list1, list2, threshold=0.83):
    # Trích xuất Description từ từng đối tượng trong danh sách `list1` và `list2`
    descriptions1 = [item.get('Description', '') for item in list1]
    descriptions2 = [item.get('Description', '') for item in list2]

    if not descriptions1 or not descriptions2:
        raise ValueError("One of the description lists is empty.")

    # Lấy embeddings cho các mô tả trong cả hai danh sách
    embeddings1 = get_embeddings(descriptions1)
    embeddings2 = get_embeddings(descriptions2)

    if embeddings1.size == 0 or embeddings2.size == 0:
        raise ValueError("One of the embedding arrays is empty.")

    # Kiểm tra xem kích thước của các embeddings có khớp không
    if embeddings1.shape[1] != embeddings2.shape[1]:
        raise ValueError("Feature dimensions of embeddings do not match.")

    # Tính toán ma trận độ tương đồng giữa các embeddings
    try:
        similarities = cosine_similarity(embeddings1, embeddings2)
    except ValueError as e:
        raise ValueError(f"Error calculating cosine similarity: {e}")

    # In ma trận độ tương đồng với các giá trị > threshold
    filtered_similarities = np.where(similarities > threshold, similarities, 0)

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
                print(f"Added pair: Index_List1={i}, Index_List2={j}, Similarity={similarity}")

    return pairs    # Trả về danh sách các cặp mô tả có độ tương đồng cao

