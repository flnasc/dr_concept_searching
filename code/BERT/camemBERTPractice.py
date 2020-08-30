from transformers import CamembertTokenizer, CamembertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity as cosine

if __name__ == '__main__':
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    camembert = CamembertModel.from_pretrained("camembert-base")
    tokenized_sentence1 = tokenizer.tokenize("l'oiseau est rouge")
    tokenized_sentence2 = tokenizer.tokenize("Il y avait une tonne de nourriture")
    encoded_sentence1 = tokenizer.encode(tokenized_sentence1)
    encoded_sentence1 = torch.tensor(encoded_sentence1).unsqueeze(0)
    encoded_sentence2 = tokenizer.encode(tokenized_sentence2)
    encoded_sentence2 = torch.tensor(encoded_sentence2).unsqueeze(0)
    embeddings1, _ = camembert(encoded_sentence1)
    embeddings2, _ = camembert(encoded_sentence2)
    a1 = embeddings1[0][0].detach().numpy()
    a2 = embeddings2[0][0].detach().numpy()
    print(a1)
    print(cosine([a1], [a2]))