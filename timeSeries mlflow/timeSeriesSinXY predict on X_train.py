import keras
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import pandas as pd
def generateSin():
    x = np.linspace(-10, 10, 800)
    y = np.sin(x)
    # + np.random.randn(*x.shape) * 0.3
    arr = [x,y]
    return x, y
def build_model():
    model = models.Sequential()
    #model.add(layers.Dense(2))

    model.add(layers.Dense(20, activation='tanh'))
    model.add(layers.Dense(8, activation='tanh'))
   # model.add(layers.Dense(3, activation='tanh'))
   
    # model.add(layers.Dense(32, activation='tanh'))
    # model.add(layers.Dense(64, activation='tanh'))
    # model.add(layers.Dense(32, activation='tanh'))
    # model.add(layers.Dense(16, activation='tanh'))
    # model.add(layers.Dense(8, activation='tanh'))
 



    #model.add(layers.Dense(2, activation='hard_tanh'))

    model.add(layers.Dense(1))
    #optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer="SGD", loss='mse', metrics=['mean_squared_error'])
    return model

if __name__ == "__main__":
    x, y = generateSin()


    xgreaterorless = np.zeros(len(y))
    xNearOne = np.zeros(len(y))
   
    for idx, item in enumerate(y):
        if (item >= 0):
            xgreaterorless[idx] = 1
        else:
            xgreaterorless[idx] = 0

        if (math.fabs(item) - 0.05 >=0.93):
            xNearOne[idx] = 1
        else:
            xNearOne[idx] = 0

    # x = x.reshape(-1, 1)
    # y = y.reshape(-1, 1)
    # xgreaterorless.reshape(-1, 1)
    # xNearOne.reshape(-1, 1)
    #print(y)
    # print(xgreaterorless)
    # print(xNearOne)



    d = {'x': x, 'xgos': xgreaterorless, 'xNearOne': xNearOne, 'y': y}
       
    data = pd.DataFrame(data = d)
    print(data)

    #X_train, x_test, Y_train, y_test = train_test_split(data[['x']], data[['y']], test_size=0.3, shuffle=False)
    X_train, x_test, Y_train, y_test = train_test_split(data[['x']], data[['y']], test_size=0.3, shuffle=False)
    #print(x, y)
    model = build_model()

    h = model.fit(X_train, Y_train, epochs=35, batch_size=2, validation_data=(X_train, Y_train))
    plt.plot(h.history['val_loss'])
    plt.show()
    #predictedY = model.predict(y_test)
    predictedY = model.predict(X_train)

    plt.subplot(211)
    plt.plot(X_train, Y_train, c = 'r')

    plt.plot(X_train, predictedY, linewidth = 1)
    #plt.plot(x_test, predictedY, linewidth = 1)
    plt.subplot(212)
    plt.plot(x, y, c = 'r')
    #plt.plot(x_test, predictedY)

    #plt.plot(x_test, predictedY)
    plt.show()