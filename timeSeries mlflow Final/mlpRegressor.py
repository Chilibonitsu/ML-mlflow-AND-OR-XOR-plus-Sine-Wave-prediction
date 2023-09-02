from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mlflow
mlflow.sklearn.autolog()
mlflow.set_experiment("mlpRegressor sklearn FINAL")
with mlflow.start_run(run_name="good run"):
    x = np.arange(0, np.pi * 2, 0.1)
    y = np.sin(x)

    # Split into Train and test
    xcop = x
    ycop = y
    xcop = xcop.reshape(-1, 1)
    ycop = ycop.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(xcop, ycop, shuffle=False)

    reg_model = MLPRegressor(hidden_layer_sizes = (40, 20, 64),  activation = "tanh", max_iter = 100, verbose=1, batch_size=8, early_stopping=True)
   


    reg_model.fit(x_train, y_train)
    y_predOnTrain = reg_model.predict(x_train)
    y_pred = reg_model.predict(x_test)
    print(x_train, y_train)

    xnext = np.arange(np.pi * 2, np.pi * 4, 0.1)
    ynext = np.sin(x)
    xnextcop = xnext
    ynextcop = ynext
    xnextcop = xnextcop.reshape(-1,1)
    ynextcop = ynextcop.reshape(-1,1)
    #print(xnextcop)
    y_pred = reg_model.predict(xcop)
    y_predOnNewData= reg_model.predict(xnextcop)

    # print(' R2 = ', r2_score(y_pred, y_test))
    # print('MAE = ', mean_absolute_error(y_pred, y_test))
    # print('MSE = ', mean_squared_error(y_pred, y_test))


    plt.plot(x, y, color='blue', linewidth=1, markersize='1', label = "training data")
    plt.plot(xnextcop, y_predOnNewData, color='green', linewidth=1, markersize='1', label = "unseen x + predictedY on x")
    plt.plot(xnext, ynext, 'm.', linewidth=1, markersize='1', label = "unseen x, y")
    plt.plot(xcop, y_pred, color='yellow', linewidth=1, markersize='1', label = "x, predictedY on x")
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.axis('tight')

    #plt.plot(xcop, y_predOnTrain, color='#456945', linewidth=1, markersize='1', label = "x, predictedY on x")
    plt.legend(loc='lower left')
    mlflow.log_figure(plt.gcf(), "original data + xtrain predicted + xtest predicted.png")


    plt.legend(loc='upper left')
    plt.show()