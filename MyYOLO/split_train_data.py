import os
import pandas as pd
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def split_train_and_val(data_dir):
    train_1w_filename_path = os.path.join('data', data_dir, 'train', 'train_1w.csv')
    val_1w_filename_path = os.path.join('data', data_dir, 'val', 'val_1w.csv')
    df_train_1w = pd.read_csv(train_1w_filename_path, sep=',', header=None)
    print(len(df_train_1w))
    df_train_1w = df_train_1w.sample(frac=1)
    train_1w_tds, train_1w_vds = train_test_split(df_train_1w, test_size=0.1, random_state=10)
    train_1w_tds.to_csv(train_1w_filename_path, index=False, header=False)
    train_1w_vds.to_csv(val_1w_filename_path, index=False, header=False)
    print(len(train_1w_tds))
    print(len(train_1w_vds))

    train_b_filename_path = os.path.join('data', data_dir, 'train', 'train_b.csv')
    val_b_filename_path = os.path.join('data', data_dir, 'val', 'val_b.csv')
    df_train_b = pd.read_csv(train_b_filename_path, sep=',', header=None)
    df_train_b = df_train_b.sample(frac=1)
    train_b_tds, train_b_vds = train_test_split(df_train_b, test_size=0.1, random_state=10)
    train_b_tds.to_csv(train_b_filename_path, index=False, header=False)
    train_b_vds.to_csv(val_b_filename_path, index=False, header=False)
    print(len(train_b_tds))
    print(len(train_b_vds))


if __name__ == '__main__':

    split_train_and_val('car')

