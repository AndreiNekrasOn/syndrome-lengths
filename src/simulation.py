import numpy as np
import traceback
from tqdm import tqdm

from codec import Hamming

def run(k, r, flip):
    codec = Hamming(n=k + r, k=k, d=4)

    message = np.random.randint(low=0, high=2, size=k)
    message = ''.join(str(i) for i in message)

    error = np.zeros(k+r, dtype=int)
    flip_idxs = flip
    error[flip_idxs] = 1

    codeword, syndrome = codec.encode(message, error)

    significant_bit = (syndrome!=0).argmax(axis=0)
    if (syndrome==0).all():
        significant_bit = r - 1 # use all
    num_bits = significant_bit + 1

    flag = True
    decoded = None
    while num_bits <= r and flag:
        try:
            decoded = codec.decode(codeword, syndrome[:num_bits])
            flag = False
        except ValueError:
            pass
        num_bits += 1

    assert decoded == message
    return num_bits, flip_idxs,

if __name__ == '__main__':
    s2bits = {}
    k, r = 1584, 12
    f = open('out.csv', 'w')
    for i in tqdm(range(1584)):
        num_bits, error = run(k, r, i)
        f.write(f'{i},{num_bits}\n')
        f.flush()
    f.close()

