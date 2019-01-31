import os
import numpy as np
import pandas as pd
import pickle

class Teacher:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_knowledge(self):
        import pickle
        data = {
            'a': [1, 2.0, 4 + 6j],
            'b': ("character string", b"byte string"),
            'c': {None, True, False}
        }
        ##############  Pickle the 'data' dictionary using the highest protocol available
        with open('data.pickle', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            ########################## pickle.HIGHEST_PROTOCOL, 这个不要也行
        print(f)

        with open('data.pickle', 'rb') as f:
            data = pickle.load(f)
        print(data)

if __name__ == '__main__':
    tea = Teacher(1)
    tea.get_knowledge()