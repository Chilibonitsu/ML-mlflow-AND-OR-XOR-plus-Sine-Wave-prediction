from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
mlflow.autolog()
mlflow.set_experiment("SAW approximate and predict new data FINAL")
def generatesaw():
    x = np.linspace(0, 50, 10000)
    sawx = x % 4.0
    y = x % 4.0 
    

    return x, sawx, y

def build_model():
    model = models.Sequential()
    #model.add(layers.Dense(2))

    model.add(layers.Dense(100, activation='tanh'))
    model.add(layers.Dense(50, activation='tanh'))
    model.add(layers.Dense(25, activation='tanh'))
   # model.add(layers.Dense(32, activation='tanh'))

    # ok result
    # model.add(layers.Dense(256, activation='tanh'))
    # model.add(layers.Dense(256, activation='tanh'))
    # model.add(layers.Dense(128, activation='tanh'))
    # model.add(layers.Dense(32, activation='tanh'))
    #model.add(layers.Dense(4, activation='tanh'))
   # model.add(layers.Dropout(rate=0.1))
    #model.add(layers.Dense(2, activation='hard_sigmoid'))

    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

if __name__ == "__main__":
    with mlflow.start_run(run_name="good run"):
        x, sawx, y = generatesaw()
        # x = x.reshape(-1, 1)
        # sawx = sawx.reshape(-1, 1)
        # y = y.reshape(-1, 1)
        
        d = {'x' : x, 'sawx': sawx, 'y' :y}
        data = pd.DataFrame(d)
        print(data)
        X_train, x_test, Y_train, y_test = train_test_split(data[['x', 'sawx']], data['y'], test_size=0.25, shuffle=False)
        print(X_train)
        model = build_model()

        # x_test = np.linspace(6, 12, 1000)
        # y_test = 2*np.saw(10*x)+ np.random.randn(*x.shape) * 0.3

        model.fit(X_train, Y_train, epochs=100, batch_size=80, validation_data=(x_test, y_test))
        #predictedY = model.predict(y_test)
        
        # переопределяем xy с новыми данными
        x = np.linspace(0, 75, 12000)
        y = x % 4.0 
        sawx = x % 4.0

        dd = {'x' : x, 'sawx': sawx, 'y' :y}
        expandedData = pd.DataFrame(dd)
        predictedY = model.predict(expandedData[['x', 'sawx']])

        plt.subplot(211)
        plt.plot(X_train['x'], Y_train, c = 'r', label = "train data")

        plt.plot(expandedData['x'], predictedY, c='b', linewidth = 1, label = "original data + new data")
        plt.legend(loc='upper left')
        plt.subplot(212)
        plt.plot(x, y, c = 'y', label = "original ALL data")
        plt.plot(expandedData['x'], predictedY, c ='m', label = "original + expanded data, predictedY")
        plt.legend(loc='upper left')
        mlflow.log_figure(plt.gcf(), "approximate + predict on new data.png")
        #plt.plot(x_test, predictedY)
        plt.show()