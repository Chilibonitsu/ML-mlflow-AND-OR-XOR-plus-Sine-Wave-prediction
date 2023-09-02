from sklearn import datasets
import warnings
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#mlflow.create_experiment("SVM Half Moon")
mlflow.sklearn.autolog()
experiment = mlflow.set_experiment("SVM Half Moon FINAL")
mlflow.set_experiment_tag("SVM Half Moon FINAL", 1)
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    #генерим луны
    data = datasets.make_moons(n_samples=1000, noise=0)
    x = data[0]
    y = data[1]
    #разбиваем на train test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    train_x = X_train
    test_x = X_test
    train_y = y_train
    test_y = y_test

    d = {'x': X_test[0], 'y': X_test[1], 'y_test': y_test[0]}

    eval_data = pd.DataFrame(data = d)
    #print(eval_data)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run(run_name="good run"):
        #model = SVC(kernel = 'linear')
        #задаем модель SVC
        model = SVC(kernel = 'rbf')
        #обучаем на тренировочных данных
        model.fit(X_train, y_train)
        #print(model.support_vectors_)
        #проверка работы модели на тестовых данных
        predicted_qualities = model.predict(X_test)
        print(len(predicted_qualities))
        print(len(y_test))
        #сигнатура в mlflow нужна для задания шейпов входных и входных данных
        signature = infer_signature(X_test, predicted_qualities)

        mlflow.sklearn.log_model(model, "model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")

        #Evaluate the logged model
        result = mlflow.evaluate(
            model_uri,
            eval_data,
            targets="y_test",
            model_type="regressor",
            evaluators=["default"],
        )

        x = X_test
        y = predicted_qualities

        #print(model.support_vectors_[:,0])
       # print(model.coef_)
        #print(predicted_qualities)
        plt.plot(x, predicted_qualities, color='r')
        mlflow.log_figure(plt.gcf(), "2D.png")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x0_list = []
        x1_list = []
        z_list = []
        for i in range(len(x)):
            x0, x1 = x[i]
            z = [predicted_qualities[i]]
            x0_list.append(x0)
            x1_list.append(x1)
            z_list.append(z)
        ax.scatter(np.array(x0_list), np.array(x1_list), y, color='b')
        ax.scatter(np.array(x0_list), np.array(x1_list), np.array(z_list),color ='m', s =2)
        ax.set_xlabel('x')
        ax.set_ylabel('x2')
        ax.set_zlabel('z')

        mlflow.log_figure(fig, "3D.png")

        #print(model.support_vectors_[:,0], 'd')
        plt.figure()
        plt.scatter(np.array(x0_list), np.array(x1_list), s = 5)
        plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1], s = 4)
        ab = plt.gca()
        
        xlim = ab.get_xlim()
        ylim = ab.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = model.decision_function(xy).reshape(XX.shape)
        
        ab.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
        plt.xlabel('x')
        plt.ylabel('x2')
        plt.ylabel('z')
        mlflow.log_figure(plt.gcf(), "Regression.png")
        plt.show()
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = model.predict(test_x)
        signature = infer_signature(test_x, predictions)
        #plt.scatter(signature[])
        plt.show()
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(mlflow.get_tracking_uri())
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                model, "model", registered_model_name="SVM Half Moon", signature=signature
            )
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)

        