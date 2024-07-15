#!/usr/bin/python env
from dataclasses import dataclass
from Levenshtein import ratio
import sys


@dataclass
class Filter():
    filename_sources: str
    filename_targets: str
    file_output_src: str
    file_output_tgt: str
    max_threshold: float = 0.99
    min_threshold: float = 0.25

    def __post_init__(self):
        self.load_files()
        self.filter_pairs()

    def load_files(self):
        def load_corpus(filepath):
            corpus = []
            with open(filepath, mode="r", encoding="utf-8") as c:
                corpus = [line.strip() for line in c]
            return corpus

        self.sources = load_corpus(self.filename_sources)
        self.targets = load_corpus(self.filename_targets)


    def filter_pairs(self):
        def write_output(path, sents):
            with open(path, 'w', encoding='utf-8-sig') as f:
                print(*sents, sep='\n', file=f)
        print(len(self.sources))
        src_filtered = []
        tgt_filtered = []
        for src, tgt in zip(self.sources, self.targets):
            lev_ratio = ratio(src.lower(), tgt.lower()) # Levenstein is applied using lowercase
            if lev_ratio <= self.max_threshold and lev_ratio >= self.min_threshold:
                src_filtered.append(src)
                tgt_filtered.append(tgt)
        write_output(self.file_output_src, src_filtered)
        write_output(self.file_output_tgt, tgt_filtered)

if __name__ == '__main__':
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    filename_output_src = sys.argv[3]
    filename_output_tgt = sys.argv[4]
    Filter(filename1,
           filename2,
           filename_output_src,
           filename_output_tgt)
