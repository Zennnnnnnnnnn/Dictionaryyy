import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Function to get embeddings from the GTE-Large model
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

# Function to perform average pooling on embeddings
def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Function to find high similarity pairs
def find_high_similarity_pairs(list1, list2, threshold=0.83):
    # Extract descriptions from both lists
    descriptions1 = [item['Description'] for item in list1]
    descriptions2 = [item['Description'] for item in list2]

    # Get embeddings for the descriptions in both lists
    embeddings1 = get_embeddings(descriptions1)
    embeddings2 = get_embeddings(descriptions2)

    # Compute the cosine similarity matrix between embeddings
    similarities = cosine_similarity(embeddings1, embeddings2)
    print("Similarity Matrix:")
    print(similarities)

    # Filter similarity matrix to show values above threshold
    print(f"Similarity Matrix (with values > {threshold}):")
    filtered_similarities = np.where(similarities > threshold, similarities, 0)
    print(filtered_similarities)

    # Initialize list to store high similarity pairs
    pairs = []

    # Create sets to track matched indices
    matched1 = set()
    matched2 = set()

    # Iterate through the similarity matrix to find high similarity pairs
    for i, row in enumerate(similarities):
        for j, similarity in enumerate(row):
            if similarity > threshold and i not in matched1 and j not in matched2:
                matched1.add(i)
                matched2.add(j)
                pairs.append({
                    'Index_List1': i,
                    'Index_List2': j,
                    'Similarity': similarity
                })
                print(f"Added pair: Index_List1={i}, Index_List2={j}, Similarity={similarity}")

    return pairs


