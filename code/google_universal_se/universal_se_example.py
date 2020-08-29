import tensorflow_hub as hub
import numpy as np
import tensorflow_text


# Some texts of different lengths.
english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
italian_sentences = ["cane", "I cuccioli sono carini.", "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane."]
japanese_sentences = ["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"]

embed = hub.load("universal-sentence-encoder-multilingual-large_3")

# Compute embeddings.
en_result = embed(english_sentences)
it_result = embed(italian_sentences)
ja_result = embed(japanese_sentences)

# Compute similarity matrix. Higher score indicates greater similarity.
similarity_matrix_en = np.inner(en_result, en_result)

if __name__ == '__main__':
    print(similarity_matrix_en);