from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
#mlflow.create_experiment("TimeSeries SAW")
mlflow.tensorflow.autolog()
# Note that only tensorflow>=2.3 are supported.
experiment = mlflow.set_experiment("TimeSeries SAW")
mlflow.set_experiment_tag("TimeSeries SAW", 1)
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))


def generateSaw():
    x = np.linspace(0, 60, 2000)
    y = x % 4.0 
    #+ np.random.randn(*x.shape) * 0.3
    arr = [x,y]
    return x, y

def build_model():
    model = models.Sequential()
    #model.add(layers.Dense(2))

    model.add(layers.Dense(32, activation='tanh'))
    model.add(layers.Dense(48, activation='tanh'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    #model.add(layers.Dense(2, activation='hard_sigmoid'))
   # optimizer = optimizers.Adam(lr = 0.01)
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])
    return model

if __name__ == "__main__":
    x, y = generateSaw()

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    print(x)
    with mlflow.start_run(run_name='xy Saw'):
        X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=False)
        xtestcop = x_test.transpose()[0]
        ytestcop = y_test.transpose()[0]

        model = build_model()

        model.fit(X_train, Y_train, epochs=80, batch_size=20,validation_data=(x_test, y_test)) 
        #predictedY = model.predict(y_test)
        predictedY = model.predict(x_test)

        signature = infer_signature(x_test, predictedY)

        mlflow.tensorflow.log_model(model, "model", signature=signature)#coding
        model_uri = mlflow.get_artifact_uri("model")
        print(model_uri)
        d = {'x': predictedY.transpose()[0], 'y': ytestcop}
       
        # eval_data = pd.DataFrame(data = d)
        # print(eval_data)
        # result = mlflow.evaluate(
        #     model_uri,
        #     eval_data,
        #     targets='y',
        #     model_type="regressor",
        #     evaluators=["default"]
        # )
   
        plt.subplot(211)
        plt.plot(X_train, Y_train, c = 'r')
       
        plt.plot(x_test, predictedY, linewidth = 1)
        mlflow.log_figure(plt.gcf(), "Train+predict.png")
        plt.subplot(212)
        plt.plot(x, y, c = 'r')
        plt.plot(x_test, predictedY)
        mlflow.log_figure(plt.gcf(), "Initial+predict.png")
        #plt.plot(x_test, predictedY)
        plt.show()