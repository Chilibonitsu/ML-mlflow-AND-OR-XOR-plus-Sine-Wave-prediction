import keras
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import pandas as pd
def generateSin():
    x = np.linspace(-10, 10, 10000)
    y = np.sin(12*x)
    # + np.random.randn(*x.shape) * 0.3
    arr = [x,y]
    return x, y
def build_model():
    model = models.Sequential()
    #model.add(layers.Dense(2))


    # model.add(layers.Dense(2, activation='relu'))
    # model.add(layers.Dense(2, activation='relu'))
    # model.add(layers.Dense(8, activation='tanh'))

    model.add(layers.Dense(256, activation='sigmoid'))
    model.add(layers.Dense(256, activation='tanh'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='elu'))
    # model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dense(128, activation='tanh'))

    #model.add(layers.Dense(6, activation='sigmoid'))



    # model.add(layers.Dense(16, activation='tanh'))
    # model.add(layers.Dense(8, activation='tanh'))
    # model.add(layers.Dense(4, activation='relu'))

    #model.add(layers.Dense(32, activation='tanh'))



   #model.add(layers.Dense(8, activation='tanh'))
    #model.add(layers.Dense(3, activation='tanh'))

    # model.add(layers.Dense(32, activation='tanh'))
    # model.add(layers.Dense(32, activation='elu'))
    # model.add(layers.Dense(32, activation='gelu'))
    # model.add(layers.Dense(28, activation='tanh'))
    # model.add(layers.Dense(16, activation='tanh'))
    #model.add(layers.Dense(8, activation='tanh'))




    #model.add(layers.Dense(2, activation='hard_tanh'))


    model.add(layers.Dense(1))
    #optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer="adam", loss='mse', metrics=['mean_squared_error'])
    return model

if __name__ == "__main__":
    x, y = generateSin()


    xgreaterorless = np.zeros(len(y))
    xNearOne = np.zeros(len(y))
    xFourParts = np.zeros(len(y))
    for idx, item in enumerate(y):
        if (item >= 0):
            xgreaterorless[idx] = 1
        else:
            xgreaterorless[idx] = 0

        if (math.fabs(item)  >=0.94):
            xNearOne[idx] = 1
        else:
            xNearOne[idx] = 0

        if (item >=0.75):
            xFourParts[idx] = 1
        elif (item >=0.5):
            xFourParts[idx] = 0.75
        elif (item  >=0.25):
            xFourParts[idx] = 0.5   
        elif (item >=0):
            xFourParts[idx] = 0.25 
        elif (item >=-0.25):
            xFourParts[idx] = 0 
        elif (item >=-0.5):
            xFourParts[idx] = -0.5 
        elif (item >=-0.75):
            xFourParts[idx] = -0.75
        elif (item >=-1):
            xFourParts[idx] = -1
    # x = x.reshape(-1, 1)
    # y = y.reshape(-1, 1)
    # xgreaterorless.reshape(-1, 1)
    # xNearOne.reshape(-1, 1)
    #print(y)
    # print(xgreaterorless)
    # print(xNearOne)


    
    d = {'x': x, 'xgos': xgreaterorless, 'xNearOne': xNearOne, 'xFourParts': xFourParts, 'y': y}
    
   #print(d['y'].values < -0.6)   
    data = pd.DataFrame(data = d)
    print(data)

   # X_train, x_test, Y_train, y_test = train_test_split(data[['x', 'xgos', 'xNearOne', 'xFourParts']], data[['y']], test_size=0.3, shuffle=False)
    X_train, x_test, Y_train, y_test = train_test_split(data[['x']], data[['y']], test_size=0.3, shuffle=False)
    # plt.plot(x_test, y_test)
    # plt.show()
    #print(x, y)
    model = build_model()

    h = model.fit(X_train, Y_train, epochs=30, batch_size=8, validation_data=(x_test, y_test))
    plt.plot(h.history['val_loss'])
    plt.show()
    #predictedY = model.predict(y_test)
    predictedY = model.predict(x_test)

    plt.subplot(211)
    plt.plot(X_train, Y_train, c = 'r')

    #plt.plot(x_test, predictedY, linewidth = 1)
    plt.plot(x_test, predictedY, linewidth = 1)
    plt.subplot(212)
    plt.plot(x, y, c = 'r')
    #plt.plot(x_test, predictedY)

    #plt.plot(x_test, predictedY)
    plt.show()