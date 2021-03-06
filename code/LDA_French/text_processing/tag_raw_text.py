"""
   Author: Dylan Hayton-Ruffner
   Description: This is the first step in our preprocessing pipeline. This role of this program is to segment the text.
   Status: Finished

"""

import csv




############################# MAIN #############################

def main():
    print("-----SEGMENT EXTRACTION-----")
    file_lines = open("../data/raw_text/soi-meme-full.txt", 'r').readlines()
    with_pb = add_page_breaks(file_lines)
    combined = combine_split(with_pb)

    with open("../data/raw_segments/soi-meme-full1.csv", "w+") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        for i in range(len(combined)):
            writer.writerow([i, combined[i]])

    return 0


############################# LOAD DATA #############################
def add_page_breaks(file_lines):
    """

    :param file_lines: list with all lines in file
    :return: a list of the file's lines with page break markers inserted
    Description: Detects page breaks in source file and add them explicitly as new lines.
    Removes footnotes from the

    """
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
        if line.endswith("?") or line.endswith("!") or line.endswith("."):
            combined.append(line)
            i += 1
        elif line != "page_break":
            if with_pb[i+1] == "page_break":
                com_count += 1
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






