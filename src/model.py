import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae

class RandomGuesser:

    def __init__(self):
        self.predictor = lambda _: 0

    def fit(self, y_train: np.ndarray):
        def predictor(_):
            return np.mean(y_train)
        self.predictor = predictor

    def predict(self, y_test):
        return np.array(list(map(self.predictor, y_test)))


if __name__ == '__main__':
    df = pd.read_csv('out.csv', header=None,
                     names=['syndrome_length', 'error_idx', 'error_vector', ''])
    X = df['error_idx'].values.reshape(-1, 1) # pyright: ignore
    y = df['syndrome_length'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    print(pd.DataFrame(y_train).describe())
    print(f'Unique classes: {len(set(y_train))}')

    baseline = RandomGuesser()
    baseline.fit(np.array(y_train))
    y_rand = baseline.predict(np.array(y_test))

    print('Random Guess MAE: {}'.format(mae(y_test, y_rand)))
    print('Random Guess MAPE: {}'.format(mape(y_test, y_rand)))


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
    # All results
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']

    print('MLP MAE: {}'.format(mae(y_test, y_pred)))
    print('MLP MAPE: {}'.format(mape(y_test, y_pred)))


