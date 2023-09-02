import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#runs = mlflow.search_runs(search_all_experiments=True)

#runs = mlflow.search_runs(experiment_names=["mlpRegressor sklearn FINAL, timeSeries betterSin FINAL"], order_by=['metrics.`my mae`'])
sklearnRuns = mlflow.search_runs(experiment_names=["mlpRegressor sklearn FINAL"], order_by=['metrics.`training_mean_squared_error`'])
tfRuns = mlflow.search_runs(experiment_names=["timeSeries betterSin FINAL"], order_by=['metrics.`mean_squared_error`'])

#print(runs.iloc[14]['metrics.my rmse'])
print(tfRuns.columns)
bestSklearnMetric = sklearnRuns.iloc[0]['metrics.training_mean_squared_error']
bestTfMetric = tfRuns.iloc[0]['metrics.mean_squared_error']

print("MLP Sklearn mse: ", bestSklearnMetric)
print("TF: ", bestTfMetric)

x = np.arange(0, np.pi * 2, 0.1)
y = np.sin(x)
xnext = np.arange(np.pi * 2, np.pi * 4, 0.1)
ynext = np.sin(x)

print(tfRuns.iloc[0]['artifact_uri'])
model = mlflow.pyfunc.load_model(f"runs:/{tfRuns.iloc[0]['run_id']}/Model")

y_pred = model.predict(xnext)
y_predOnTrain = model.predict(x)

plt.plot(x, y, color='blue', linewidth=1, markersize='1', label = "training data")
plt.plot(xnext, y_pred, color='green', linewidth=1, markersize='1', label = "test x, predictedY")
plt.plot(xnext, ynext, 'm.', linewidth=1, markersize='1', label = "test x, y")
plt.plot(x, y_predOnTrain, color='yellow', linewidth=1, markersize='1', label = "x, predictedY on x")
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.legend(loc='upper left')
plt.show()
#runstime = pd.to_datetime(runs['end_time'], format = '%Y-%m-%d %H:%M')
#plt.scatter(runs['run_id'], runs['metrics.mae'])

 