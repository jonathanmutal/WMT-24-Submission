#!/usr/bin/python env

from dataclasses import dataclass
from Levenshtein import ratio
import sys


@dataclass
class Filter():
    filename_sources: str
    filename_targets: str
    max_threshold: float = 0.99
    min_threshold: float = 0.25

    def __post_init__(self):
        self.load_files()
        self.calculate_distance()
        
    def load_files(self):
        def load_corpus(filepath):
            corpus = []
            with open(filepath, mode="r", encoding="utf-8") as c:
                corpus = [line.strip() for line in c]
            return corpus
            
        self.sources = load_corpus(self.filename_sources)
        self.targets = load_corpus(self.filename_targets)


    def calculate_distance(self):
        print(len(self.sources))
        outName = ".".join(self.filename_sources.split(".")[0:2])
        outFile = open(f"{outName}.levenstein", "w", encoding="utf-8")
        for src, tgt in zip(self.sources, self.targets):
            lev_ratio = ratio(src.lower(), tgt.lower()) # Levenstein is applied using lowercase
            if lev_ratio <= self.max_threshold and lev_ratio >= self.min_threshold:
            	print(src, tgt, sep="\t", file=outFile)
        outFile.close()


if __name__ == '__main__':
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    Metrics(filename1, filename2)