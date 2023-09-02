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
    y = np.sin(6*x)
    # + np.random.randn(*x.shape) * 0.3
    arr = [x,y]
    return x, y
def build_model():
    model = models.Sequential()

    model.add(layers.Dense(20, activation = 'tanh',use_bias = True))
    model.add(layers.Dense(20, activation = 'tanh',use_bias = True))
    model.add(layers.Dense(1, activation = 'tanh',use_bias = True))

    model.add(layers.Dense(1))
    optimizer = keras.optimizers.Adam(lr=0.005)
    model.compile(optimizer="SGD", loss='mse', metrics=['mean_squared_error'])
    return model

def currentXfeature(Y_train, idx):
    #x,y тут тестовые
    #X_train['period'][idx] = X_train['x'][idx] % 12
    if (Y_train.iloc[[idx]]['y'].item() >=0.75):
        return 1
    elif (Y_train.iloc[[idx]]['y'].item() >=0.5):
        return 0.75
    elif (Y_train.iloc[[idx]]['y'].item()>=0.25):
        return 0.5
    elif (Y_train.iloc[[idx]]['y'].item()>=0):
        return 0.25
    elif (Y_train.iloc[[idx]]['y'].item()>=-0.25):
        return 0
    elif (Y_train.iloc[[idx]]['y'].item()>=-0.5):
        return -0.25
    elif (Y_train.iloc[[idx]]['y'].item()>=-0.75):
        return -0.5
    elif (Y_train.iloc[[idx]]['y'].item() >=-1):
        return -0.75

if __name__ == "__main__":
    x, y = generateSin()


    predYparts = np.zeros(len(y))

    period = np.zeros(len(y))
   

    # x = x.reshape(-1, 1)
    # y = y.reshape(-1, 1)
    # predYparts.reshape(-1, 1)
    # x3.reshape(-1, 1)
    #print(y)
    # print(predYparts)
    # print(x3)


    
    d = {'x': x, 'predYparts': predYparts, 'period': period, 'y': y}
    
   
    data = pd.DataFrame(data = d)
    #print(data)

   # X_train, x_test, Y_train, y_test = train_test_split(data[['x', 'xgos', 'x3', 'period']], data[['y']], test_size=0.3, shuffle=False)
    X_train, x_test, Y_train, y_test = train_test_split(data[['x', 'predYparts', 'period']], data[['y']], test_size=0.3, shuffle=False)

    coun = 0
    for idx, item in enumerate(X_train['x'], 1):
        #X_train['period'][idx] = X_train['x'][idx] % 12

        coun = idx
        if idx == X_train.shape[0]:
            break
        #X_train.iloc[[idx]]['period'] = X_train.iloc[[idx]]['x'] % 12
        period[idx] = X_train.iloc[[idx]]['x'] % 12
        #Y_train.iloc[[idx]]['y'].item() >=0.75
        if (Y_train.iloc[[idx-1]]['y'].item() >=0.75):
            predYparts[idx] = 1
            # data['predYparts'][idx] = 1
            # X_train['predYparts'][idx] = 1

 
        elif (Y_train.iloc[[idx-1]]['y'].item() >=0.5):
            predYparts[idx] = 0.75
  
            
        elif (Y_train.iloc[[idx-1]]['y'].item() >=0.25):
            predYparts[idx-1] = 0.5  

        elif (Y_train.iloc[[idx-1]]['y'].item() >=0):
            predYparts[idx] = 0.25 

        elif (Y_train.iloc[[idx-1]]['y'].item() >=-0.25):
            predYparts[idx] = 0 
      

        elif (Y_train.iloc[[idx-1]]['y'].item() >=-0.5):
            predYparts[idx] = -0.25 

        elif (Y_train.iloc[[idx-1]]['y'].item()>=-0.75):
            predYparts[idx] = -0.5

        elif (Y_train.iloc[[idx-1]]['y'].item() >=-1):
            predYparts[idx] = -0.75

    print(coun)
    X_train['predYparts'] = predYparts[:X_train.shape[0]]
    X_train['period'] = period[:X_train.shape[0]]
    # plt.plot(x_test, y_test)
    # plt.show()
    #print(x, y)
    model = build_model()
    
    X_train = X_train[1:]
    Y_train = Y_train[1:]
    #plt.plot(h.history['val_loss'])
   # plt.show()


    print("TRAIN", X_train, Y_train)
    h = model.fit(X_train, Y_train, epochs=100, batch_size=100, validation_data=(x_test, y_test))

    # x_test['predYparts'][x_test.index[0]] = currentXfeature(Y_train, len(Y_train)-1)
    # x_test['period'][x_test.index[0]] = x_test['x'][x_test.index[0]] % 12
    x_test.at[x_test.index[0], 'predYparts'] = currentXfeature(Y_train, len(Y_train)-1)
    x_test.at[x_test.index[0], 'period'] = x_test['x'][x_test.index[0]] % 12
 #   print(x_test['predYparts'] )

    
    firstPrediction = model.predict(pd.DataFrame(x_test.iloc[0, :]).T)
    #predictedY = model.predict(y_test)
   # a = np.array(firstPrediction)
   # allPredictedY = np.append(a, firstPrediction[0])
    
    #dy = {'y', firstPrediction}
    #dataYpred  = pd.DataFrame(data = dy)
    dataYpred  = pd.DataFrame(firstPrediction, columns=['y'])
    print(dataYpred.shape[0])
    
    for idx in range(1, x_test.shape[0]-2980):
        predYparts[idx+7000] = currentXfeature(dataYpred, idx-1)
        period[idx+7000] = x_test['x'][x_test.index[0]+idx] % 12
        x_test.at[idx+7000, 'predYparts'] = predYparts[idx+7000]
        x_test.at[idx+7000, 'period'] = period[idx+7000]

        predictedY = model.predict(pd.DataFrame(x_test.iloc[idx]).T)
        dataYpred.loc[idx] = predictedY[0]
        print(idx, "IDX")



    print('TEST', x_test,y_test)
    #plt.plot(x_test[:20], dataYpred)
       # allPredictedY = np.append(allPredictedY, predictedY)

    # for idx, item in enumerate(x_test, 1):
    #     print(item)
    #     x_test['predYparts'][idx] = currentXfeature(dataYpred, idx-1)
    #     x_test['period'][idx] = x_test['x'][x_test[0]] % 12

    #     predictedY = model.predict(item)
    #     dataYpred['y'][idx] = predictedY 
    #     allPredictedY = np.append(allPredictedY, predictedY)


    print(dataYpred)
    plt.plot(x_test.iloc[:20, 0], dataYpred)
    plt.show()
    # plt.subplot(211)
    # plt.plot(X_train, Y_train, c = 'r')

    # #plt.plot(x_test, predictedY, linewidth = 1)
    # plt.plot(x_test, predictedY, linewidth = 1)
    # plt.subplot(212)
    # plt.plot(x, y, c = 'r')
    # #plt.plot(x_test, predictedY)

    # #plt.plot(x_test, predictedY)
    # plt.show()