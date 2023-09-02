from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
import shap

def generateSaw():
    x = np.linspace(-1, 1, 100)
    y = x % 4.0 
    #+ np.random.randn(*x.shape) * 0.3
    arr = [x,y]
    return x, y



x, y = generateSaw()
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


model = mlflow.pyfunc.load_model("models:/SAW shap train testing 1/1")
#shap_data = mlflow.shap.load_explainer("models:/Saw-test/1")
#shap.plots.scatter(shap_data)
predictedY = model.predict(y)

plt.plot(x, y, c = 'r', linewidth = 4)
plt.plot(x, predictedY, c = 'b', linewidth = 1)

plt.show()