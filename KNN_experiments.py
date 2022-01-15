import subprocess

import numpy as np

from KNN import KNNClassifier
from utils import *

target_attribute = 'Outcome'


def run_knn(k, x_train, y_train, x_test, y_test, formatted_print=True):
    neigh = KNNClassifier(k=k)
    neigh.train(x_train, y_train)
    y_pred = neigh.predict(x_test)
    acc = accuracy(y_test, y_pred)
    print(f'{acc * 100:.2f}%' if formatted_print else acc)


def get_top_b_features(x, y, b=5, k=51):
    """
    :param k: Number of nearest neighbors.
    :param x: array-like of shape (n_samples, n_features).
    :param y: array-like of shape (n_samples,).
    :param b: number of features to be selected.
    :param method: sklearn method
    :return: indices of top 'b' features as the result of selection/dimensionality reduction on sample
            sets using sklearn.feature_selection module
    """
    # TODO: Implement get_top_b_features function
    #   - Note: The brute force approach which examines all subsets of size `b` will not be accepted.

    assert 0 < b < x.shape[1], f'm should be 0 < b <= n_features = {x.shape[1]}; got b={b}.'
    x_copy = x.copy()
    top_b_features_indices = []
    # ====== YOUR CODE: ======
    n_samples = x.shape[0]
    for a in range(b):
        max_acc = -np.inf
        best_feature_index = None
        for j in range(x_copy.shape[1]):
            array = []
            if j not in top_b_features_indices:
                top_b_features_indices.append(j)
                for r in range(10):
                    x = x_copy[:, top_b_features_indices]
                    indices = np.arange(x.shape[0])
                    np.random.shuffle(indices)
                    x = np.array(x[indices])
                    y = np.array(y[indices])
                    a = int(0.8 * n_samples)
                    x_train1 = x[:a]
                    y_train1 = y[:a]
                    x_valid = x[a:]
                    y_valid = y[a:]
                    neigh = KNNClassifier(k=k)
                    neigh.train(x_train1, y_train1)
                    y_pred = neigh.predict(x_valid)
                    acc = accuracy(y_valid, y_pred)
                    array.append(acc)
                mean_acc = np.mean(array)
                if mean_acc > max_acc:
                    max_acc = mean_acc
                    best_feature_index = j
                top_b_features_indices.remove(j)
        top_b_features_indices.append(best_feature_index)
    return top_b_features_indices


def delete_feature(x_train, x_test, index):
    np.delete(x_train, index, 1)
    np.delete(x_test, index, 1)


def insert_feature(x_train, x_test, index, train_x_col, test_x_col):
    np.insert(x_train, index, train_x_col, 1)
    np.insert(x_test, index, test_x_col, 1)



def run_cross_validation():
    """
    cross validation experiment, k_choices = [1, 5, 11, 21, 31, 51, 131, 201]
    """
    file_path = str(pathlib.Path(__file__).parent.absolute().joinpath("KNN_CV.pyc"))
    subprocess.run(['python', file_path])


def exp_print(to_print):
    print(to_print + ' ' * (30 - len(to_print)), end='')


# ========================================================================
if __name__ == '__main__':
    """
       Usages helper:
       (*) cross validation experiment
            To run the cross validation experiment over the K,Threshold hyper-parameters
            uncomment below code and run it
    """
    # run_cross_validation()

    # # ========================================================================

    attributes_names, train_dataset, test_dataset = load_data_set('KNN')
    x_train, y_train, x_test, y_test = get_dataset_split(train_set=train_dataset,
                                                         test_set=test_dataset,
                                                         target_attribute='Outcome')

    best_k = 51
    b = 5

    # # ========================================================================
    print("-" * 10 + f'k  = {best_k}' + "-" * 10)
    exp_print('KNN in raw data: ')
    run_knn(best_k, x_train, y_train, x_test, y_test)

    top_m = get_top_b_features(x_train, y_train, b=5, k=best_k)
    x_train_new = x_train[:, top_m]
    x_test_test = x_test[:, top_m]
    print(top_m)
    exp_print(f'KNN in selected feature data: ')
    run_knn(best_k, x_train_new, y_train, x_test_test, y_test)
