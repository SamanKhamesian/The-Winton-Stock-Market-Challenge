import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from Source.config import FILE_PATH, INPUT_SIZE, MAX_SIZE, SPLIT_RATIO


class Driver:
    def __init__(self):
        self.x_train, self.y_train, self.x_test, self.y_test = self.__load_data()

    def __load_data(self):
        df = pd.read_csv(FILE_PATH + 'train.csv').astype(float)
        df.fillna(df.mean(axis=0), inplace=True)
        msk = np.random.rand(df.shape[0]) < SPLIT_RATIO

        X = df.iloc[:, 0:INPUT_SIZE]
        Y = df.iloc[:, INPUT_SIZE:MAX_SIZE]

        x_train = self.__scale_data(data=X[msk])
        x_test = self.__scale_data(X[~msk])
        y_train = Y[msk]
        y_test = Y[~msk]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def __scale_data(data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
