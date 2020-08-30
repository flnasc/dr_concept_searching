from sentence_transformers import SentenceTransformer, util

if __name__ == '__main__':
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

    # Two lists of sentences
    sentences1 = ['la chouette est orange',"il n'y a pas beaucoup de nourriture"]

    sentences2 = ["la chouette est jaune","il y a tellement de nourriture"]

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Output the pairs with their score
    for i in range(len(sentences1)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))