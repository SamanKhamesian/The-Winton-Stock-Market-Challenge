from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential

from Source.config import INPUT_SIZE


class Model:
    def __init__(self):
        self.__model = Sequential()
        self.__model.add(Dense(48, input_dim=INPUT_SIZE, kernel_initializer='normal'))
        self.__model.add(Dense(24, activation='relu'))
        self.__model.add(Dense(12, activation='relu'))
        self.__model.add(Dense(6, activation='relu'))
        self.__model.add(Dense(1, activation='linear'))
        self.__model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        self.mcp_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    def fit(self, x_train, y_train):
        self.__model.fit(x_train, y_train, epochs=10, batch_size=50, callbacks=[self.early_stopping, self.mcp_save],
                         verbose=0)

    def predict(self, x_test):
        return self.__model.predict(x_test)
