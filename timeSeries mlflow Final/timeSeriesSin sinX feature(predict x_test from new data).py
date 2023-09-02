from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
mlflow.set_experiment("sinX predict on x_test new data FINAL")
def generateSin():
    x = np.linspace(0, 8, 1000)
    y = 2*np.sin(10*x) + np.random.randn(*x.shape) * 0.3
    sinx = 2*np.sin(10*x)
    #+ np.random.randn(*x.shape) * 0.3
    #arr = [x, sinx, y]
    return x, sinx, y
def build_model():
    model = models.Sequential()
    #model.add(layers.Dense(2))

    model.add(layers.Dense(256, activation='tanh'))
    model.add(layers.Dense(512, activation='tanh'))
    model.add(layers.Dense(16, activation='tanh'))
    #model.add(layers.Dense(4, activation='tanh'))
   # model.add(layers.Dropout(rate=0.1))
    #model.add(layers.Dense(2, activation='hard_sigmoid'))

    model.add(layers.Dense(1))
    model.compile(optimizer='SGD', loss='mse', metrics=['mse'])
    return model

if __name__ == "__main__":
    with mlflow.start_run(run_name="good run"):
        x, sinx, y = generateSin()
        # x = x.reshape(-1, 1)
        # sinx = sinx.reshape(-1, 1)
        # y = y.reshape(-1, 1)
        
        d = {'x' : x, 'sinx': sinx, 'y' :y}
        data = pd.DataFrame(d)
        print(data)
        X_train, x_test, Y_train, y_test = train_test_split(data[['x', 'sinx']], data['y'], test_size=0.25, shuffle=False)
        print(X_train)
        model = build_model()

        # x_test = np.linspace(6, 12, 1000)
        # y_test = 2*np.sin(10*x)+ np.random.randn(*x.shape) * 0.3

        model.fit(X_train, Y_train, epochs=100, batch_size=40, validation_data=(x_test, y_test))
        #predictedY = model.predict(y_test)
        predictedY = model.predict(x_test)

        plt.subplot(211)
        plt.plot(X_train['x'], Y_train, c = 'r', label = "xy train")

        plt.plot(x_test['x'], predictedY, linewidth = 1, label = "xtest predictedY")
        plt.legend()
        plt.subplot(212)
        plt.plot(x, y, c = 'r')
        #plt.plot(x_test['x'], predictedY)
        plt.legend()
        mlflow.log_figure(plt.gcf(), "prediction on new data - x_test.png")
        #plt.plot(x_test, predictedY)
        plt.show()