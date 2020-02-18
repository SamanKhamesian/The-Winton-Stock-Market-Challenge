import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

from Source.config import DAILY_RETURN_COL, INTRA_DAY_COL
from Source.config import VAL
from Source.driver import Driver
from Source.model import Model

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def show_results(figure_title, error_title, plot_res, res):
    error = mse(res['x'], res['y'])
    print('MSE for ' + error_title + ' is:', error)

    plt.plot(range(0, VAL), plot_res['x'][:VAL], 'b-', label='Predicted prices')
    plt.plot(range(0, VAL), plot_res['y'][:VAL], 'r--', label='Actual prices')
    plt.xlabel('First 200 rows of data')
    plt.ylabel('Stock Value')
    plt.title(figure_title)
    plt.legend()
    plt.show()


def run():
    d = Driver()

    model = Model()
    run_model(model, d.x_train, d.y_train, d.x_test, d.y_test, DAILY_RETURN_COL)
    run_model(model, d.x_train, d.y_train, d.x_test, d.y_test, INTRA_DAY_COL)


def run_model(model, x_train, y_train, x_test, y_test, col_name):
    figure_title = ''
    error_title = ''
    plot_res, res = pd.DataFrame(), pd.DataFrame()
    for col in col_name:
        model.fit(x_train=x_train, y_train=y_train[col])

        if col == 'Ret_121' or col == 'Ret_PlusOne':
            figure_title = col
            if col is 'Ret_121':
                error_title = 'Intra-Day'
            else:
                error_title = 'D_1 and D_2'
            plot_res = plot_res.append(pd.DataFrame({'x': model.predict(x_test).flatten().tolist(), 'y': y_test[col].tolist()}))
        res = res.append(pd.DataFrame({'x': model.predict(x_test).flatten().tolist(), 'y': y_test[col].tolist()}))

    show_results(figure_title, error_title, plot_res, res)


if __name__ == '__main__':
    run()
