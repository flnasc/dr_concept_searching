"""
   Author: Dylan Hayton-Ruffner
   Description: Parses terminal output from Mallet and builds a graph plotting iterations vs perplexity

           Usage: <path to mallet output>
           Description: Takes one command line argument that specifies the path to the file containing mallets output.
           To use, run gensim_topic_model.py, copy and paste the output into a text file and then run from the command
           line: python3 graph_mallet_perplexity.py <path_to_output>

   Status: Finished
   ToDo: N/A

   NOTES: Concept path and results path are hard-coded


"""


import matplotlib.pyplot as plt
import re
import sys

REGEX_EXPR = r"<\d+> LL/token: -[\d\.]+"
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: <mallet_output_path>")
        exit(1)
    file_text = open(sys.argv[1]).read()
    results = re.findall(REGEX_EXPR, file_text)
    graph_iters = []
    graph_perplex = []
    for line in results:
        iteration = line.split()[0]
        iteration = iteration.replace("<", "")
        iteration = iteration.replace(">", "")
        perplexity = line.split()[-1]
        graph_iters.append(int(iteration))
        graph_perplex.append(float(perplexity))

    plt.plot(graph_iters, graph_perplex, 'gs-')
    plt.title("Likelihood Per Token vs Iterations on SM with 18 Topics")
    plt.ylabel("Likelihood Per Token")
    plt.xlabel("Iterations")

    plt.show()





