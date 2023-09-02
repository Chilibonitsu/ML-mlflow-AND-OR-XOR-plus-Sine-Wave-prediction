from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
def generateSin():
    # x = np.linspace(0, 8, 1000)
    # y = 2*np.sin(10*x) + np.random.randn(*x.shape) * 0.3
    # sinx = 2*np.sin(10*x) + np.random.randn(*x.shape) * 0.3


    x = np.linspace(0, 50, 2000)
    sinx = x % 4.0
    y = x % 4.0 
    

    return x, sinx, y
def build_model():
    model = models.Sequential()
    #model.add(layers.Dense(2))

    model.add(layers.Dense(256, activation='tanh'))
    model.add(layers.Dense(128, activation='tanh'))
    model.add(layers.Dense(64, activation='tanh'))
    #model.add(layers.Dense(4, activation='tanh'))
   # model.add(layers.Dropout(rate=0.1))
    #model.add(layers.Dense(2, activation='hard_sigmoid'))

    model.add(layers.Dense(1))
    model.compile(optimizer='SGD', loss='mse', metrics=['mse'])
    return model

if __name__ == "__main__":
    x, sinx, y = generateSin()
    # x = x.reshape(-1, 1)
    # sinx = sinx.reshape(-1, 1)
    # y = y.reshape(-1, 1)
    
    d = {'x' : x, 'sax': sinx, 'y' :y}
    data = pd.DataFrame(d)
    print(data)
    X_train, x_test, Y_train, y_test = train_test_split(data[['x', 'sax']], data['y'], test_size=0.25, shuffle=False)
    print(X_train)
    model = build_model()

    # x_test = np.linspace(6, 12, 1000)
    # y_test = 2*np.sin(10*x)+ np.random.randn(*x.shape) * 0.3

    model.fit(X_train, Y_train, epochs=50, batch_size=40, validation_data=(x_test, y_test))
    #predictedY = model.predict(y_test)
    predictedY = model.predict(x_test)

    plt.subplot(211)
    plt.plot(X_train['x'], Y_train, c = 'r')

    plt.plot(x_test['x'], predictedY, linewidth = 1)
    plt.subplot(212)
    plt.plot(x, y, c = 'r')
    plt.plot(x_test['x'], predictedY)

    #plt.plot(x_test, predictedY)
    plt.show()