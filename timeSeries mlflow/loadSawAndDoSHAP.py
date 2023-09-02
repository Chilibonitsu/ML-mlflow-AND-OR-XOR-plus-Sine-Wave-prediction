from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
import shap
mlflow.tensorflow.autolog()
# Note that only tensorflow>=2.3 are supported.
experiment = mlflow.set_experiment("TimeSeries SAW")
mlflow.set_experiment_tag("TimeSeries SAW", 1)
# print("Experiment_id: {}".format(experiment.experiment_id))
# print("Artifact Location: {}".format(experiment.artifact_location))
# print("Tags: {}".format(experiment.tags))

def generateSaw():
    x = np.linspace(-500, 500, 700)
    y = x % 4.0 
    #+ np.random.randn(*x.shape) * 0.3
    arr = [x,y]
    return x, y



x, y = generateSaw()
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

with mlflow.start_run(run_name="ne zapusk"):
    model = mlflow.pyfunc.load_model("models:/hsnck wo shap run/1")
    #print(model.model_uri)

    X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)
    predictedY = model.predict(y_test)
    ytestcop = y_test.transpose()[0]

    signature = infer_signature(y_test, predictedY)

    model_uri = mlflow.get_artifact_uri("model")
    print(model_uri)
    d = {'x': predictedY.transpose()[0], 'y': ytestcop}
    eval_data = pd.DataFrame(data = d)

    print(eval_data)
    result = mlflow.evaluate(
        model,
        eval_data,
        targets='y',
        model_type="regressor",
        evaluators=["default"]
    )

    # ...
    plt.plot(x, y, c = 'r', linewidth = 4)
    plt.plot(x_test, predictedY, c = 'b', linewidth = 1)

    plt.show()