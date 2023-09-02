from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def generateSin():
    x = np.linspace(0, 6, 10000)
    y = 2*np.sin(10*x) + np.random.randn(*x.shape) * 0.3
    arr = [x,y]
    return x, y
def build_model():
    model = models.Sequential()
    #model.add(layers.Dense(2))

    model.add(layers.Dense(30, activation='tanh'))
    model.add(layers.Dense(8, activation='tanh'))
    #model.add(layers.Dense(2, activation='hard_sigmoid'))

    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

if __name__ == "__main__":
    x, y = generateSin()
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=False)
   
    model = build_model()

    model.fit(Y_train, Y_train, epochs=5, batch_size=8)
    #predictedY = model.predict(y_test)
    predictedY = model.predict(y_test)

    plt.subplot(211)
    plt.plot(X_train, Y_train, c = 'r')

    plt.plot(x_test, predictedY, linewidth = 1)
    plt.subplot(212)
    plt.plot(x, y, c = 'r')
    plt.plot(x_test, predictedY)

    #plt.plot(x_test, predictedY)
    plt.show()