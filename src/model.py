import argparse
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor


class RandomGuesser:

    def __init__(self):
        self.predictor = lambda _: 0

    def fit(self, y_train: np.ndarray):
        def predictor(_):
            return np.mean(y_train)
        self.predictor = predictor

    def predict(self, y_test):
        return np.array(list(map(self.predictor, y_test)))

def print_metrics(name, y_test, y_pred):
    print(f'{name} MAE: {mae(y_test, y_pred)}')
    print(f'{name} MAPE: {mape(y_test, y_pred)}')


def predict_baseline(X_train, X_test, y_train, y_test):
    baseline = RandomGuesser()
    baseline.fit(np.array(y_train))
    y_rand = baseline.predict(np.array(y_test))
    print_metrics("RandomGuesser", y_test, y_rand)
    return y_rand


def find_optimal_params(X_train, X_test, y_train, y_test):
    parameter_space = {
        'hidden_layer_sizes': [(50, 20), (10, 2), (50,50,50), (50,100,50), (100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'learning_rate': ['constant','adaptive'],
        }
    clf = MLPRegressor(max_iter=500, verbose=True)
    clf = GridSearchCV(clf, parameter_space, n_jobs=4, cv=3, verbose=3)
    # clf = CatBoostRegressor(verbose=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if clf.best_params_:
        print('Best parameters found:\n', clf.best_params_)
    print_metrics("MLP", y_test, y_pred)

def predict_best_model(X_train, X_test, y_train, y_test):
    clf = MLPRegressor(max_iter=500, verbose=True,
                       hidden_layer_sizes=(100,),
                       activation='relu',
                       alpha=0.0001,
                       learning_rate='constant',
                       solver='adam'
                       )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_metrics("MLP Best", y_test, y_pred)
    return y_pred

def save_prediction(X_test, y_test, y_pred, errors):
    # print(errors[0])
    entropy = list(map(calc_entropy, errors))
    assert len(y_test) == len(y_pred) and len(y_pred) == len(entropy)
    with open('result.csv', 'w') as f:
        f.write(f'real,prediction,entropy\n')
        for i in range(len(y_test)):
            f.write(f'{y_test[i]},{int(y_pred[i])},{entropy[i]}\n')


def calc_entropy(error: str):
    err = np.array(list(error), dtype=int)
    p = 0.
    for i in err:
        p += (i != 0)
    p /= len(err)
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def prepare_data(filename):
    df = pd.read_csv(filename, header=None,
                     names=['syndrome_length', 'weight', 'error', 'empty'])
    X = df.drop(['syndrome_length','empty'], axis=1)
    print(X.head())
    y = df['syndrome_length'].values
    return train_test_split(X, y, test_size=0.1, random_state=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', action='store_true')
    args = parser.parse_args()

    df_train, df_test, y_train, y_test = prepare_data('out.csv')
    print(pd.DataFrame(y_train).describe())
    print(f'Unique classes: {len(set(y_train))}')
    X_train = df_train['weight'].values.reshape((-1,1)) # pyright: ignore
    X_test = df_test['weight'].values.reshape((-1,1)) # pyright: ignore
    predict_baseline(X_train, X_test, y_train, y_test)
    if args.g:
        find_optimal_params(X_train, X_test, y_train, y_test)
    else:
        y_pred = predict_best_model(X_train, X_test, y_train, y_test)
        save_prediction(X_test, y_test, y_pred, df_test['error'].values)

