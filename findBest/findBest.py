import mlflow

run = mlflow.search_experiments()
print(run[0])