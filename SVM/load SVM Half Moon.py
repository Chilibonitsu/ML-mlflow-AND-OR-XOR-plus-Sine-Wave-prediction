import mlflow
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

model = mlflow.pyfunc.load_model("models:/for loadSVM 2/1")

data = datasets.make_moons(n_samples=1000, noise=0.05)
x = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
train_x = X_train
test_x = X_test
train_y = y_train
test_y = y_test

d = {'x': X_test[0], 'y': X_test[1], 'y_test': y_test[0]}

eval_data = pd.DataFrame(data = d)
#mlflow.pyfunc.get_model_dependencies("models:/for loadSVM 2/1")
#shap_data = mlflow.shap.load_explainer("models:/for loadSVM 2/1")
#shap.plots.scatter(shap_data)

predicted_qualities = model.predict(X_test)

print(predicted_qualities)