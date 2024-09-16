import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

class RandomGuesser:

    def __init__(self):
        self.predictor = lambda _: 0

    def fit(self, y_train: np.ndarray):
        def predictor(_):
            p = len(y_train[y_train==12])/len(y_train)
            if p > np.random.random():
                return 12
            return 11
        self.predictor = predictor

    def predict(self, y_test):
        return np.array(list(map(self.predictor, y_test)))


if __name__ == '__main__':
    df = pd.read_csv('out.csv', header=None,
                     names=['error_idx', 'syndrome_length'])
    X = df['error_idx'].values.reshape(-1, 1) # pyright: ignore
    y = df['syndrome_length'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    baseline = RandomGuesser()
    baseline.fit(np.array(y_train))
    y_rand = baseline.predict(np.array(y_test))

    print('Random Guess Accuracy: {}'.format(np.mean(y_rand == y_test)))


    parameter_space = {
        'hidden_layer_sizes': [(50, 20), (10, 2), (50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
        }
    clf = MLPClassifier(max_iter=100)
    clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Best parameters found:\n', clf.best_params_)
    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    print('MLP Accuracy: {}'.format(np.mean(y_pred == y_test)))

