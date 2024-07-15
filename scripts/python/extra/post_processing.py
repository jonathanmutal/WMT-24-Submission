import sys

def main(args):
    with open(args[0], 'r') as f:
        src = [line.strip() for line in f.readlines()]
    with open(args[1], 'r') as f:

        tgt = [line.strip() for line in f.readlines()]
    new_tgt = []
    for s, t in zip(src, tgt):
        if s.endswith('.') and not t.endswith('.'):
            print(s)
            print(t)
            t += '.'
        new_tgt.append(t)

    with open(args[1] + '.pos' , 'w') as f:
        print(*new_tgt, sep='\n', file=f)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <src> <tgt>")
        exit()
    main(sys.argv[1:])
