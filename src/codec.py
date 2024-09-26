from functools import cache
import numpy as np

class Hamming:

    def __init__(self, n: int, k: int, d: int = 3) -> None:
        self.n = n
        self.k = k
        self.d = d
        self.H = self.build_check_matrix()
        self.G = self.build_gen_matrix()
        self.syndromes = self.get_syndromes()

    def _build_check_matrix3(self, n: int, k: int):
        r = n - k
        if n > 2**r - 1:
            raise ValueError(f"Impossible code: ({n}, {k})")
        H_1 = np.zeros(shape=(r, k), dtype=int)
        num = 3
        for j in range(k):
            for i in range(r):
                H_1[i][k - j - 1] = (num >> i) & 1
            num += 1
            if num & (num - 1) == 0: # skip powers of 2
                num += 1
        H_2 = np.identity(n=r, dtype=int)
        return np.hstack([H_1, H_2])

    def build_check_matrix(self):
        if self.d != 3 and self.d != 4:
            raise ValueError("Minimum distance should be 3 or 4")
        if self.d == 3:
            return self._build_check_matrix3(self.n, self.k)
        r = self.n - self.k
        H3 = self._build_check_matrix3(self.n, self.k + 1)
        H4 = np.vstack([np.ones(self.n, dtype=int), H3])
        H4[0] = H4.sum(axis=0) % 2
        for i in range(1, r):
            if H4[i][self.k] != 0:
                H4[i] += H4[0]
        H4 = H4 % 2
        return H4

    def build_gen_matrix(self):
        H1 = self.H[:,:self.k]
        G = np.hstack([np.identity(self.k, dtype=int), H1.T])
        return G

    def get_syndromes(self):
        H = self.build_check_matrix()
        syndromes = H.T
        return syndromes

    def encode(self, message: str):
        if len(message) != self.k:
            raise(ValueError(f'lengths dont match {len(message)} != {self.k}'))
        m = np.array(list(message), dtype=int)
        c = (m @ self.G) % 2
        return c

    def decode(self, codeword: np.ndarray):
        '''
        Expects <= 2 errors in the codeword
        '''
        syndromes = self.syndromes
        syndrome = (codeword @ self.H.T) % 2
        errors = np.identity(self.n, dtype=int)
        j = 0
        if (syndrome!=np.zeros_like(syndrome)).any():
            j = -1
            for i in range(self.n):
                if (syndromes[i]==syndrome).all():
                    j = i
            if j != -1:
                codeword = (codeword + errors[j]) % 2
        if j == -1:
            return '', True
        m = codeword[:self.k] % 2
        return ''.join(str(int(i)) for i in m), False

