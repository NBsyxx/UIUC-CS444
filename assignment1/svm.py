"""Support Vector Machine (SVM) model."""

import numpy as np
import copy
import random


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me we don't do reg_term here
        grad_w = np.array([[0 for j in range(X_train.shape[1])] for i in range(self.n_class)]).astype("float")
        N,D = X_train.shape
        
        for i in range(N):
            tmp_grad = np.array([[0 for j in range(X_train.shape[1])] for i in range(self.n_class)]).astype("float")
            sum_xi = np.array([0 for i in range(D)]).astype("float")
            for c in range(len(tmp_grad)):
                if c != y_train[i]: # 
                    if  np.dot(self.w[y_train[i]],X_train[i]) - np.dot(self.w[c],X_train[i]) < 1:
                        tmp_grad[c] = X_train[i]
                        tmp_grad[y_train[i]] -= X_train[i]
            grad_w += tmp_grad
        return grad_w
        
        
            
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        BATCH_SIZE = 128
        
        # start with random weights
        random.seed(666)
        b_up = np.min(X_train)
        b_low = np.max(X_train)
        self.w = np.array([[random.uniform(b_low,b_up) for j in range(X_train.shape[1])] for i in range(self.n_class)])
        
        N,D = X_train.shape
        it = N//BATCH_SIZE
        
        pred_svm_t = self.predict(X_train)
        t_acc = self.get_acc(pred_svm_t, y_train)
        
        for epoch in range(self.epochs):
            print("epoch",epoch)
            for i in range(N//BATCH_SIZE): # feed in training data batch-wise
                X_train_batch = X_train[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
                y_train_batch = y_train[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
        
                grad_w = self.calc_gradient(X_train_batch, y_train_batch)
                old_w = copy.deepcopy(self.w)
                for c in range(len(self.w)):
                    # TODO: update w
                    self.w[c] = (1-self.lr*self.reg_const/it)*old_w[c] - self.lr*grad_w[c]
                
                ret = self.predict(X_train)
                t_cur_acc = self.get_acc(ret, y_train)
                pred_svm_v = self.predict(X_val)
                cur_v_acc = self.get_acc(pred_svm_v, y_val)
                if i%1 == 0:
                    print("\tBatch",i,"of",it,"training acc",t_cur_acc,"val acc",cur_v_acc)
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
        # TODO: implement me
        ret = []
        for i in range(X_test.shape[0]):
            scores = np.dot(self.w, X_test[i])
#             print(scores)
            max_score = float("-inf")
            max_class = -1
            for j in range(len(scores)):
                if scores[j] > max_score:
                    max_score = scores[j]
                    max_class = j
            ret.append(max_class)
        return np.array(ret)
