from typing import List, Literal
import numpy as np
import traceback
from tqdm import tqdm

from codec import Hamming


IS_DEBUG = 0


def debug(*values):
    global IS_DEBUG
    if IS_DEBUG:
        print(values)

def recursive_decode(msg: str,
                     error: np.ndarray) -> tuple[str, int]:
    if len(msg) == 0:
        raise ValueError("recursive_decode: empty_arg[msg]")
    codec = get_codec(len(msg))
    r = codec.n - codec.k
    # debug('Codec used: ', codec.n, codec.k)
    c_hat = codec.encode(msg)
    # debug(len(c_hat), len(msg), r)
    # debug(ndarr2str(error))
    # debug(ndarr2str(np.pad(error, (0, r))))
    c_hat += np.pad(error, (0, r)) # c_hat = (msg|syndrome)
    c_hat = c_hat % 2
    # debug(f'error sum: {error.sum()}')
    if sum(error) > 2:
        decoded, err = '', True
    else:
        decoded, err = codec.decode(c_hat)
    del codec
    if not err:
        return decoded, r
    half = len(msg) // 2
    decoded_left, left = recursive_decode(msg[:half], error[:half])
    decoded_right, right = recursive_decode(msg[half:], error[half:])
    return decoded_left + decoded_right, left + right

def ndarr2str(arr: np.ndarray) -> str:
    return ''.join(str(i) for i in arr)

def run(k: int, error: np.ndarray) -> int:
    message = np.random.randint(low=0, high=2, size=k)
    # debug(f'error sum: {sum(error)}')
    slen = 0
    decoded_message, slen = recursive_decode(ndarr2str(message), error)
    if decoded_message != ndarr2str(message):
        raise(ValueError(f'Decoding error: \nGot: {decoded_message}\nMsg: {ndarr2str(message)}'))

    return slen

def generate_error(k: int, weight: int | None = None) -> np.ndarray:
    error = np.zeros(shape=(k,), dtype=int)
    if weight != None:
        num_errors = weight
    else:
        num_errors = np.random.randint(0, high=k)
    error_positions = np.random.default_rng()\
            .choice(k, num_errors, replace=False)
    error[error_positions] = 1
    return error


cached_codecs = {}
def get_codec(msg_len: int) -> Hamming:
    global cached_codecs
    if msg_len in cached_codecs:
        return cached_codecs[msg_len]
    s_len = 0
    while msg_len + s_len >= 2**s_len - 1:
        s_len += 1
    codec = Hamming(n=msg_len + s_len+1, k=msg_len, d=4)
    cached_codecs[msg_len] = codec
    return codec


if __name__ == '__main__':
    k = 1584
    f = open('out.csv', 'w')
    for i in tqdm(range(10*k)):
        error = generate_error(k)
        slen = run(k, error)
        f.write(f'{slen},{sum(error)},{ndarr2str(error)},\n')
    f.close()

