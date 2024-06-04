from dataclasses import dataclass
from Levenshtein import distance
import sys


@dataclass
class Metrics():
    filename_predictions: str
    filename_references: str

    def __post_init__(self):
        self.load_files()

        print("Levenshtein distance: {}".format(self.calculate_distance()))


    def load_files(self):
        def load_corpus(filepath):
            corpus = []
            with open(filepath, mode="r") as c:
                corpus = [line.strip() for line in c]
            return corpus
        self.predictions = load_corpus(self.filename_predictions)
        self.references = load_corpus(self.filename_references)

    def calculate_distance(self):
        return sum([Levenshtein.distance(pred, ref) for pred, ref in zip(self.predictions, self.references)]) / len(self.predictions)

if __name__ == '__main__':
    print(Levenshtein.distance("hola", "holas"))
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    Metrics(filename1, filename2)