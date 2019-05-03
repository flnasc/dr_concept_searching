import csv
import sys
class ConvHeuristic:

    def __init__(self, path_a, path_b):
        self.path_a = path_a
        self.path_b = path_b
        self.data_a = self.read_csv(self.path_a)
        self.data_b = self.read_csv(self.path_b)


    def conv_heuristic(self):
        total = 0
        pairs = []
        for i in range(len(self.data_a)):
            row_a = self.data_a[i]
            min = len(row_a)
            pair = [i,-1,-1]
            for j in range(len(self.data_b)):
                row_b = self.data_b[j]
                dif = self.row_difference(row_a, row_b)
                if dif < min:
                    min = dif
                    pair[1] = j
            total += min
            pair[2] = min
            pairs.append(pair)
        return total, pairs


    def row_difference(self, row_a, row_b):
        dif = 0
        for item in row_a:
            if item not in row_b:
                dif += 1
        return dif


    def read_csv(self, path):
        with open(path) as csv_file:
            reader = csv.reader(csv_file)
            file_rows = []
            for row in reader:
                file_rows.append(row[1:30])

            return file_rows

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: path_to_file_a path_to_file_b")
    ch = ConvHeuristic(sys.argv[1], sys.argv[2])
    total_dif, row_matches = ch.conv_heuristic()
    print("Total Difference:", total_dif)
    print("Row Matches:", row_matches)
