import numpy as np
import unittest

from src.codec import Hamming

class UHammingTest(unittest.TestCase):
    def test_incorrect_input(self):
        self.assertRaises(Exception, lambda _ : Hamming(7, 7))
        self.assertRaises(Exception, lambda _ : Hamming(7, 4, 4))
        def long_msg():
            codec = Hamming(7, 3)
            message = '1111'
            codec.encode(message)
        self.assertRaises(Exception, long_msg)

    def test_excess_errors(self):
        codec = Hamming(7, 3, 4)
        message = '101'
        error = np.array([1, 1, 0, 0, 0, 0, 0])
        codeword = codec.encode(message)
        self.assertRaises(Exception, lambda _: codec.decode((codeword + error) % 2)[0])

    def test_no_errors(self):
        codec = Hamming(7, 4)
        message = '1111'
        mhat = codec.decode((codec.encode(message)) % 2)[0]
        self.assertEqual(mhat, message)

        codec = Hamming(7, 3, 4)
        message = '000'
        mhat = codec.decode((codec.encode(message)) % 2)[0]
        self.assertEqual(mhat, message)

    def test_all_errors3(self):
        codec = Hamming(7, 4)
        message = '1010'
        errors = np.identity(7)
        for error in errors:
            codeword = codec.encode(message)
            codeword = (codeword + error) % 2
            mhat = codec.decode(codeword)[0]
            self.assertEqual(mhat, message, msg=f'Error: {error}')

    def test_all_errors4(self):
        codec = Hamming(7, 3, 4)
        message = '100'
        errors = np.identity(7)
        for error in errors:
            codeword = codec.encode(message)
            codeword = (codeword + error) % 2
            mhat = codec.decode(codeword)[0]
            self.assertEqual(mhat, message)

if __name__ == '__main__':
    unittest.main()

