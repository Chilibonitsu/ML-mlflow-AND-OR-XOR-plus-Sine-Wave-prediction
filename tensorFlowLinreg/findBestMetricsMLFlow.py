import mlflow
import pandas as pd
import matplotlib.pyplot as plt
#runs = mlflow.search_runs(search_all_experiments=True)

runs = mlflow.search_runs(experiment_names=["tensorFlowLinreg AND"], order_by=['metrics.`my mae`'])

print(runs.iloc[14]['metrics.my rmse'])
print(runs['end_time'])
runstime = pd.to_datetime(runs['end_time'], format = '%Y-%m-%d %H:%M')
plt.scatter(runs['run_id'], runs['metrics.mae'])
plt.show()
#for i in runs:
#https://www.databricks.com/blog/2020/02/18/how-to-display-model-metrics-in-dashboards-using-the-mlflow-search-api.html
#print(runs)