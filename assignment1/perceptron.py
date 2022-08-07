# multiclass perceptron

"""Perceptron model."""

import numpy as np
import random
import copy


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this W
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val=X_train , y_val=y_train):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        
        # TODO: implement me
        N,D = X_train.shape
        random.seed(5)
        upper_b = np.min(X_train)
        lower_b = np.max(X_train)
        self.w = np.array([[random.uniform(lower_b,upper_b) for j in range(X_train.shape[1])] for i in range(self.n_class)])

        
        pred_svm_t = self.predict(X_train)
        t_acc = self.get_acc(pred_svm_t, y_train)
        
        for epoch in range(self.epochs):
            # for each of the training data
            for i in range(X_train.shape[0]):
                for c in range(len(self.w)):
                    if np.dot(self.w[c],X_train[i]) > np.dot(self.w[y_train[i]],X_train[i]):
                        self.w[y_train[i]] = self.w[y_train[i]] + self.lr*X_train[i]
                        self.w[c] = self.w[c] - self.lr*X_train[i]
                if i%100 == 0:
                    ret = self.predict(X_train)
                    t_cur_acc = self.get_acc(ret, y_train)
                    pred_svm_v = self.predict(X_val)
                    cur_v_acc = self.get_acc(pred_svm_v, y_val)

                    print("\tBatch",i,"of",N,"training acc",t_cur_acc,"val acc",cur_v_acc)
                    # early stop
                    if cur_v_acc >= 83: # found 83
                            return

        
    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100
    
                        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        ret = []
        for i in range(X_test.shape[0]):
            scores = np.dot(self.w, X_test[i])
            max_score = float("-inf")
            max_class = -1
            for i in range(len(scores)):
                if scores[i] > max_score:
                    max_score = scores[i]
                    max_class = i
            ret.append(max_class)
        return np.array(ret)
