"""
   Author: Dylan Hayton-Ruffner
   Description: Takes a csv file as input and outputs a new csv file with all the segments lemmatized.
   Status: Unfinished
   ToDo: N/A

"""

import spacy
import fr_core_news_sm
import csv
import sys
import random

BOOK = "symb-du-mal-full"


def main():
    if len(sys.argv) < 3:
        print("Usage: source_path dest_path")
        quit(1)

    # load spacy model
    print("loading model...")
    nlp = spacy.load("fr_core_news_sm")

    # load segments from csv (delimited with ("|")
    print("loading segments from " + sys.argv[1])
    segs = load_segs_from_csv(sys.argv[1])

    # lemmatize each segment
    print("lemmatizing segs...")
    lemma_segs = []

    for i in range(len(segs)):

        # analyze each segment and add it to the fully lemmatized set
        lemma_segs.append(analyze_seg(segs[i], nlp))
        sys.stdout.write("\r%i segments processed" % i)
        sys.stdout.flush()

    # write to csv file
    print("\nwriting segs...")
    write_to_csv(sys.argv[2] , lemma_segs)


def load_segs_from_csv(path):
    """
        Parameters: path -> str, path to csv file
        Return: segs -> list, a list of all segments
    """
    segs = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        for row in reader:
            segs.append(row[1])
    return segs


def analyze_seg(seg, nlp):
    """

    :param seg: string
    :param nlp: spacy "fr_core_news_sm" model
    :return: lemmatized segment
    """
    # get a list of the natural language (includeing POS) info for each word
    doc = nlp(seg)

    # create a new segment by joining all the lemmas
    new_doc = " ".join([word.lemma_ for word in doc])
    return new_doc


def write_to_csv(write_path, segs):
    """

    :param write_path: string, csvfile path
    :param segs: list, a list of all segments
    :return: void
    """
    with open(write_path, "w+") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        for i in range(len(segs)):
            writer.writerow([i,segs[i]])


if __name__ == "__main__":
    main()