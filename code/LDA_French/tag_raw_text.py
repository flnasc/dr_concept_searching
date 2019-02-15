"""
   Author: Dylan Hayton-Ruffner
   Description: This is the first step in our preprocessing pipeline. This role of this program is to segment the text.

            Query: If the concept-word exsists in the top 4 words of a topic, all the paragraphs associated with that topic and have
            the concept word are returned

            After each successful query, the results are formated into an excel file and written to the results folder.

   Status: Finished
   NOTES: Concept path and results path are hard-coded


"""

import csv




############################# MAIN #############################

def main():
    print("-----SEGMENT EXTRACTION-----")
    file_lines = open("../../data/soi-meme-full.txt", 'r').readlines()
    with_pb = add_page_breaks(file_lines)
    combined = combine_split(with_pb)

    with open("../../data/soi-meme-full.csv", "w+") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        for i in range(len(combined)):
            writer.writerow([i, combined[i]])

    return 0


############################# LOAD DATA #############################
def add_page_breaks(file_lines):
    with_pb = ["start."]
    num_footnotes = 0
    num_blanklines = 0
    num_pagebreaks = 0
    for line in file_lines:
        if '\f' in line and with_pb[-1] != "page_break":
            num_pagebreaks += 1
            with_pb.append("page_break")
        if line.strip() == "":
            num_blanklines += 1
            continue
        elif len(line.strip().split()) > 10 and not line[0].isdigit():
            with_pb.append(line.replace("\n", "").strip())
        elif line[0].isdigit():
            num_footnotes += 1

    print("-----LOCATING PAGE BREAKS-----")
    print("Ignored", num_footnotes, "footnotes")
    print("Ignored", num_blanklines, "blank lines")
    print("Found", num_pagebreaks, "page breaks")
    print("Returned", len(with_pb) - num_pagebreaks, "segments")
    return with_pb

def combine_split(with_pb):
    combined = []
    i = 0
    com_count = 0
    annomaly = 0
    while i < (len(with_pb) - 3):
        line = with_pb[i].strip()
        if line.endswith("?") or line.endswith("!") or line.endswith(".") or line.endswith("»"):
            combined.append(line)
            i += 1
        elif line != "page_break":
            if with_pb[i+1] == "page_break":
                com_count += 1
                # print("combined:", with_pb[i])
                # print("with:", with_pb[i + 2])
                combined.append((with_pb[i] + with_pb[i + 2]).strip())
                i += 3
            else:
                annomaly += 1
                combined.append(with_pb[i])
                i += 1
        else:
            i += 1
            pass

    print("-----COMBING SPLIT PARAGRAPHS-----")
    print("Combined", com_count, "segments")
    print("Added", annomaly, "segments that did not fit the algorithm")
    print("Returned", len(combined), "segments")
    return combined


if __name__ == "__main__":
    main()






