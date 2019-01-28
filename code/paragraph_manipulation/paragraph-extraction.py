""" 
   Author: Dylan Hayton-Ruffner
   Description: Runs lda on the given corpus, prints out resulting topics and queries every concept from the concept file.

            Query: If the concept-word exsists in the top 4 words of a topic, all the paragraphs associated with that topic and have
            the concept word are returned

            After each successful query, the results are formated into an excel file and written to the results folder.
   
   Status: Finished
   ToDo: N/A

   NOTES: Concept path and results path are hard-coded


"""
TOPIC_PRESSENCE_THRESHOLD = 0.3
REGEX_PATTERN = u'(?u)\\b\\w\\w\\w\\w+\\b'
MIN_WORD_COUNT = 10
NUM_TOPICS = 10
TOP_N_SEGS=10
TOP_N_WORDS=0
MIN_DF = 0.00
MAX_DF = 1.00
FILETYPE = 'xml'
CONCEPTS_PATH = "../../data/concepts.txt"
import bs4 as Soup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

from nltk import word_tokenize  
from nltk import pos_tag          
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import csv


class LemmaTokenizer(object):
     def __init__(self):
        self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in re.findall(REGEX_PATTERN, doc)]



############################# MAIN #############################

def main():
    print("\n-----LDA CONCEPT DETECITON-----")
    text_corpus, text_corpus_ids, raw_corpus, raw_corpus_ids, filepath = load_corpus('v')
    print(len(text_corpus_ids))
    with open("paragraphs_soe.csv", "w+") as csvfile:
        writer = csv.writer(csvfile, delimiter="@")
        for i in range(len(text_corpus)):
            writer.writerow([text_corpus_ids[i], text_corpus[i].strip().replace("\n", "")])

    return 0



############################# LOAD DATA ############################# 

def load_document(filepath):
    """
        Description:Opens and loads the file specified by filepath as a raw txt string; assumes valid text file format.
        Input: String -> filepath of file from current directory
        Output: Entire contents of text file as a string

    """

    #assert(filepath.endswith(".txt")), "Function: Load Document -> File specificed by filepath is not of type .txt"
    file = open(filepath, 'r')
    file_string = file.read()
    file.close()

    return file_string


def load_corpus(v):
    """html and xml are supported"""

    #get text data from file as a raw string, parse with bs4 and extract paragraph tags -> list of bs4.element.Tag objects
    # filepath = input("Filepath to corpus: ")
    filepath = "../../data/symbolism-of-evil.xml"
    if v:
        print("LOADING FILE: " + filepath)

    doc_string = load_document(filepath)
    doc_soup = Soup.BeautifulSoup(doc_string, FILETYPE)
    doc_para = doc_soup.find_all('p') #use beautiful soup to find all contents of the paragraph

    #get contents of each paragraph tag and add them to the list 'corpus'
    raw_corpus = []
    raw_corpus_ids = []
    cleaned_corpus = []
    cleaned_corpus_ids = []
    vectorizer = CountVectorizer(stop_words = 'english', lowercase= True)

    for i in range(0, len(doc_para)):
        raw_corpus.append(doc_para[i].get_text())
        raw_corpus_ids.append(doc_para[i].get("ID"))
        #use vectorizer to count number of significant words in each paragraph
        try:
            vectorizer.fit_transform([doc_para[i].get_text()])
            matrix = vectorizer.transform([doc_para[i].get_text()])

            if matrix.sum() > MIN_WORD_COUNT:
                cleaned_corpus.append(doc_para[i].get_text())
                cleaned_corpus_ids.append(doc_para[i].get("ID"))

            else:
                continue
        except ValueError:
            continue

    return cleaned_corpus, cleaned_corpus_ids, raw_corpus, raw_corpus_ids, filepath




if __name__ == "__main__":
    main()






