
import warnings
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging
#настраиваем логирование mlflow
#пишем autolog для sklearn
mlflow.sklearn.autolog()
#задаем эксперимент
experiment = mlflow.set_experiment("AND linreg FINAL")
mlflow.set_experiment_tag("AND linreg FINAL", 1)
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    #задаем таблицу истинности AND
    data = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])

    targets = np.array([
        [0.],
        [0.],
        [0.],
        [1.]
    ])


    x = data
    y = targets

    
    train_x = x
    test_x = x
    train_y = y
    test_y = y

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    #чтобы логались метрики и параметры модели пишем start_run
    with mlflow.start_run(run_name="good run"):
        #задаем и обучаем модель
        lr = ElasticNet(alpha=0, l1_ratio=1)
        lr.fit(train_x, train_y)

        #
        predicted_qualities = lr.predict(test_x)
        print(predicted_qualities)

        plt.plot(x, predicted_qualities, color='r')
        #mlflow.log_figure(plt.gcf(), "2d x, prediction.png")
        ax = plt.figure().add_subplot(projection='3d')

        x0_list = []
        x1_list = []
        z_list = []
        for i in range(len(x)):
            x0, x1 = x[i]
            z = [predicted_qualities[i]]
            x0_list.append(x0)
            x1_list.append(x1)
            z_list.append(z)
        ax.scatter(np.array(x0_list), np.array(x1_list), y, color='r', label ='original data')
        ax.scatter(np.array(x0_list), np.array(x1_list), np.array(z_list), label ='predicted data')
        plt.legend()

        plt.xlabel('x')
        plt.ylabel('x2')
        plt.ylabel('y')
        ax.set_xlabel('x')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        mlflow.log_figure(plt.gcf(), "3D original data, prediction.png")
        plt.show()
    
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        
        # mlflow.log_param("alpha", alpha)
        # mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = lr.predict(test_x)
        signature = infer_signature(test_x, predictions)
        #plt.scatter(signature[])
        #plt.show()
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)

        