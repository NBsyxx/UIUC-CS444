"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1/(1+ np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        print(X_train.shape)
        random.seed(5)
        upper_b = np.min(X_train)
        lower_b = np.max(X_train)
        self.w = np.zeros(X_train.shape[1])
        print("w",self.w)
        
        for epoch in range(self.epochs):
            cnt = 0
            # for each of the training data
            for i in range(X_train.shape[0]):
                  grad_w = X_train[i] * (y_train[i] - self.sigmoid(np.dot(self.w, X_train[i])))
                  self.w += self.lr*grad_w
            print("\t","epoch",epoch,"training")
        print(self.w)
            
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
        # TODO: implement me
        ret = []
        for i in range(len(X_test)):
            score = np.dot(self.w,X_test[i])
#             print("!!!",score,)
#             print(X_test[i])
#             print("score",score)
#             print(self.sigmoid(score))
            if self.sigmoid(score) >= threshold:
                ret.append(1)
            else:
                ret.append(0)
#         print(ret)
        return np.array(ret)