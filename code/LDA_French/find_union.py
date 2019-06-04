import csv


class UnionInfo:

    def __init__(self, file_path_a, file_path_b):
        self.file_path_a = file_path_a
        self.file_path_b = file_path_b
        self.file_a_concepts = self.get_concepts(file_path_a)
        self.file_b_concepts = self.get_concepts(file_path_b)
        self.analysis = Analysis(self.file_a_concepts, self.file_b_concepts)

    def get_concepts(self, file_path):
        concepts = []
        with open(file_path) as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:
                concept = Concept(row[0])
                if len(row) > 1:
                    concept.segments = row[1:]
                concepts.append(concept)

        return concepts

    def print(self):
        print(self.analysis)


class Concept:

    def __init__(self, title):
        self.title = title
        self.segments = []


class Analysis:

    def __init__(self, concepts_a, concepts_b):
        self.concepts_a = concepts_a
        self.concepts_b = concepts_b
        self.results = self.analyze()

    def __str__(self):
        string = "Analysis\n"
        for line in self.results:
            string += str(line)
        return string

    def analyze(self):
        data = []
        if len(self.concepts_a) != len(self.concepts_b):
            print("# Concepts extracted from input file do not match!")
            return
        for i in range(len(self.concepts_a)):
            concept_a = self.concepts_a[i]
            concept_b = self.concepts_b[i]
            if concept_a.title != concept_b.title:
                print("Concepts from files are not the same!")
                return

            union_count = 0
            non_union_count = 0
            for segment in concept_a.segments:
                if segment in concept_b.segments:
                    union_count += 1
                else:
                    non_union_count += 1

            data.append([concept_a.title, union_count, non_union_count])

        return data


if __name__ == "__main__":
    ui = UnionInfo("../../../data/concepts_sm.csv", "../../../data/concepts_sm1.csv")
    ui.print()



