from keras import models
from keras import layers
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
    x = np.linspace(-500, 500, 700)
    y = x % 4.0 
    #+ np.random.randn(*x.shape) * 0.3
    arr = [x,y]
    return x, y

def build_model():
    model = models.Sequential()
    #model.add(layers.Dense(2))

    model.add(layers.Dense(30, activation='hard_sigmoid'))
    model.add(layers.Dense(8, activation='hard_sigmoid'))
    #model.add(layers.Dense(2, activation='hard_sigmoid'))

    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    x, y = generateSaw()

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    print(x)
    with mlflow.start_run():
        X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)
        xtestcop = x.transpose()[0]
        ytestcop = y.transpose()[0]

        model = build_model()

        model.fit(Y_train, Y_train, epochs=60, batch_size=8) 
        #predictedY = model.predict(y_test)
        predictedY = model.predict(y_test)

        signature = infer_signature(x_test, predictedY)

        mlflow.tensorflow.log_model(model, "model", signature=signature)#coding
        model_uri = mlflow.get_artifact_uri("model")
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
        #mlflow.log_figure(plt.gcf(), "Train+predict.png")
        plt.subplot(212)
        plt.plot(x, y, c = 'r')
        plt.plot(x_test, predictedY)
        #mlflow.log_figure(plt.gcf(), "Initial+predict.png")
        #plt.plot(x_test, predictedY)
        plt.show()