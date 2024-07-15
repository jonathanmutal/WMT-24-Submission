
from transformers import AutoTokenizer
import sys


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args[0])
    tokenizer.src_lang = str(args[2])
    lines = []
    print(args[1])
    with open(args[1], 'r') as f:
        lines = f.readlines()
    max_token = max([len(tokenizer.tokenize(src.strip())) for src in lines])
    print("\n".join([str(len(tokenizer.tokenize(src.strip()))) for src in lines]))
    print(max_token)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <model name> <file> <lang>")
        exit()
    main(sys.argv[1:])
