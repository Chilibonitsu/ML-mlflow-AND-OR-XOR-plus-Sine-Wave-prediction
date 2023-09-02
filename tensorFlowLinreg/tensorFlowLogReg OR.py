from keras import models
from keras import layers
import mlflow
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
from keras.optimizers import Adam 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
tf.random.set_seed(1234)
mlflow.autolog()
mlflow.set_experiment("TF LogReg OR FINAL")
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def build_model(train):
    model = models.Sequential()
   

    #model.add(layers.Dense(2))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.add(layers.Dense(1))
    optimizer = Adam(learning_rate=0.05)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    return model

def cross_validation(train):
    
    k = 4
    num_val_samples = len(train) // k
    num_epochs = 10000
    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
        val_data = train[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y_train[i * num_val_samples:
        (i + 1) * num_val_samples]
        partial_train_data = np.concatenate(
        [train[:i * num_val_samples],
        train[(i + 1) * num_val_samples:]],
        axis=0)
        partial_train_targets = np.concatenate(
        [y_train[:i * num_val_samples],
        y_train[(i + 1) * num_val_samples:]],
        axis=0)
        model = build_model(train)
        history = model.fit(partial_train_data, partial_train_targets,
        validation_data=(val_data, val_targets),
        epochs=num_epochs, batch_size=1, verbose=0)
        mae_history = history.history['mae']
        all_mae_histories.append(mae_history)
    average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    return average_mae_history

if __name__ == "__main__":
    with mlflow.start_run(run_name="good run"):
        
        #mlflow.autolog()
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
            [1]
        ])
        x = data
        y = targets

    # print(dataWithoutY)
        # with mlflow.start_run():

        #     mlflow.set_tag("mlflow.runName", "AND TF")

        train= x
        test_x = x
        y_train = y
        test_y = y
        
        
        model = build_model(train)
        model.fit(x, y, epochs = 200)
        predicted_qualities = model.predict(test_x)
        (rmse, mae, r2) = eval_metrics(y, predicted_qualities)
        print("Predicted Y: ", predicted_qualities)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        mlflow.log_metrics({'rmse': rmse, 'mae': mae, 'r2': r2})
        #plt.plot(x, predicted_qualities, color='r')
        
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
        ax.scatter(np.array(x0_list), np.array(x1_list), y, color='r', label = "x1, x2, y")
        ax.scatter(np.array(x0_list), np.array(x1_list), np.array(z_list), color = 'g', label = "x1, x2, Ypredicted")
        
        #ax.plot_surface(np.array(x0_list), np.array(x1_list), np.array(z_list))
        
    #    ax.plot(x, predicted_qualities, zs=0, zdir = 'y')
        #5 plt.plot(x, predicted_qualities, color='r')
        
        plt.xlabel('x')
        plt.ylabel('x2')
        ax.set_xlabel('x')
        ax.set_ylabel('x2')
        ax.set_zlabel('z')
        plt.legend()
        mlflow.log_figure(plt.gcf(), "original data, predicted data.png")
        plt.show()