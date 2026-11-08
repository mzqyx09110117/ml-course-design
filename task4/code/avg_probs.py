import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for i in range(1, 21):
        acc = None
        n = 0
        for d in args.dirs:
            p = os.path.join(d, f'{i}.npy')
            if os.path.exists(p):
                v = np.load(p).astype(np.float32)
                acc = v if acc is None else acc + v
                n += 1
        if acc is not None:
            np.save(os.path.join(args.out, f'{i}.npy'), (acc / n).astype(np.float16))
    print('averaged', len(args.dirs), 'dirs ->', args.out)


if __name__ == '__main__':
    main()
