import matplotlib.lines as mlines
import warnings
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from sklearn.svm import SVC
import logging
mlflow.autolog()
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
mlflow.set_experiment("SVM XOR FINAL")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def newline(p1, p2, color=None): # функция отрисовки линии
    #function kredits to: https://fooobar.com/questions/626491/how-to-draw-a-line-with-matplotlib
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax], color=color)
    ax.add_line(l)
    return l

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL

    data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    targets = np.array([
        [0],
        [1],
        [1],
        [0]
    ])
    x = data
    y = targets


    train_x = x
    test_x = x
    train_y = y
    test_y = y


    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run(run_name="good run"):
        #model = SVC(kernel = 'linear')
        model = SVC(kernel = 'rbf')
        model.fit(x, y)
        #print(model.support_vectors_)
        predicted_qualities = model.predict(x)
        #print(model.support_vectors_[:,0])
       # print(model.coef_)
        print(predicted_qualities, "PREDICT")
        plt.plot(x, predicted_qualities, color='r')
        
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
        ax.scatter(np.array(x0_list), np.array(x1_list), y, color='b')
        ax.scatter(np.array(x0_list), np.array(x1_list), np.array(z_list),color ='r', s =2)
        ax.set_xlabel('x')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')



        print(model.support_vectors_[:,0], 'd')
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
        mlflow.log_figure(plt.gcf(), "asd.png")
        plt.show()
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        #print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
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

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                model, "model", registered_model_name="SVM XOR", signature=signature
            )
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)

        