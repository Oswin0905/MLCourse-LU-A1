############ CODE BLOCK 0 ################
# ^ DO NOT CHANGE THIS LINE

# You are not allowed to add additional imports!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, ensemble, metrics, svm, model_selection, linear_model

############ CODE BLOCK 1 ################
# ^ DO NOT CHANGE THIS LINE
def training_test_split(X, y, test_size=0.3, random_state=None):
    """ Split the features X and labels y into training and test features and labels.

    `split` indicates the fraction (rounded down) that should go to the test set.

    `random_state` allows to set a random seed to make the split reproducible.
    If `random_state` is None, then no random seed will be set.

    """

    if random_state is not None:
        np.random.seed(random_state)

    n = X.shape[0]

    indices = np.random.permutation(n)

    n_test = int(n * test_size)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test

############ CODE BLOCK 2 ################
# ^ DO NOT CHANGE THIS LINE

def true_positives(true_labels, predicted_labels, positive_class):
    pos_true = true_labels == positive_class  # compare each true label with the positive class
    pos_predicted = predicted_labels == positive_class # compare each predicted label to the positive class
    match = pos_true & pos_predicted # use logical AND (that's the `&`) to find elements that are True in both arrays
    return np.sum(match)  # count them

############ CODE BLOCK 3 ################
# ^ DO NOT CHANGE THIS LINE

def false_positives(true_labels, predicted_labels, positive_class):
    pos_predicted = predicted_labels == positive_class  # predicted to be positive class
    neg_true = true_labels != positive_class  # actually negative class
    match = pos_predicted & neg_true  # The `&` is element-wise logical AND
    return np.sum(match)  # count the number of matches

############ CODE BLOCK 4 ################
# ^ DO NOT CHANGE THIS LINE

def true_negatives(true_labels, predicted_labels, positive_class):
    neg_true = true_labels != positive_class
    neg_predicted = predicted_labels != positive_class
    match = neg_true & neg_predicted
    return np.sum(match)

def false_negatives(true_labels, predicted_labels, positive_class):
    pos_true = true_labels == positive_class
    neg_predicted = predicted_labels != positive_class
    match = pos_true & neg_predicted
    return np.sum(match)

############ CODE BLOCK 5 ################
# ^ DO NOT CHANGE THIS LINE

def precision(true_labels, predicted_labels, positive_class):
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    return TP / (TP + FP)

############ CODE BLOCK 6 ################
# ^ DO NOT CHANGE THIS LINE

def recall(true_labels, predicted_labels, positive_class):
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)
    return TP / (TP + FN)

############ CODE BLOCK 7 ################
# ^ DO NOT CHANGE THIS LINE

def accuracy(true_labels, predicted_labels, positive_class):
    # positive_class not needed for accuracy, but kept for signature consistency
    correct = (true_labels == predicted_labels)
    return np.sum(correct) / len(true_labels)

############ CODE BLOCK 8 ################
# ^ DO NOT CHANGE THIS LINE

def specificity(true_labels, predicted_labels, positive_class):
    neg_true = true_labels != positive_class
    neg_predicted = predicted_labels != positive_class
    match = neg_true & neg_predicted
    TN = np.sum(match)

    FP = np.sum((predicted_labels == positive_class) & (true_labels != positive_class))

    return TN / (TN + FP)

############ CODE BLOCK 9 ################
# ^ DO NOT CHANGE THIS LINE

def balanced_accuracy(true_labels, predicted_labels, positive_class):
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)

    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return (recall + specificity) / 2

############ CODE BLOCK 10 ################
# ^ DO NOT CHANGE THIS LINE

def F1(true_labels, predicted_labels, positive_class):
    p = precision(true_labels, predicted_labels, positive_class)
    r = recall(true_labels, predicted_labels, positive_class)

    return 2 * (p * r) / (p + r)

