import numpy as np
from sklearn.linear_model import LogisticRegression
import time

from cuml.linear_model import LogisticRegression
import cudf

def main():

    # create low rank data
    n = 400
    d = 8192

    X_train = np.random.normal(size=(n,d))
    X_test = np.random.normal(size=(n, d))
    
    y_train = (np.where(X_train[:, 0] > 0, 1, 0))
    y_test = (np.where(X_test[:, 0] > 0, 1, 0))

    print(X_train.shape, y_train.shape)

    start = time.time()
    model = LogisticRegression(C=1, fit_intercept=False)
    model.fit(X_train, y_train)
    end = time.time()
    print("Training time: ", end - start)    

    start = time.time()
    X_cudf = cudf.DataFrame(X_train)
    y_cudf = cudf.Series(y_train)

    model = LogisticRegression()
    model.fit(X_cudf, y_cudf)    
    end = time.time()
    print("Training time: ", end - start)    

if __name__ == "__main__":
    main()