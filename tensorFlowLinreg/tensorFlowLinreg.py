from keras import models
from keras import layers
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

mlflow.autolog()

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def build_model(train):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
    input_shape=(train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":

    # Read the wine-quality csv file from the URL
    data = []
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
        
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
   # y = data['quality']
    # print(data.info())
    # dataWithoutY = data.copy()
    # dataWithoutY = dataWithoutY.drop('quality', axis = 1)
   # print(dataWithoutY)
    with mlflow.start_run():
        
        #train, test = train_test_split(data)
       # print(data[0])
        #print(data[0:1, 0:1])
        #print(data[:][:-1])
        datacop = data.copy()
        train, test, y_train, y_test = train_test_split(datacop.drop(['quality'], axis = 1), data['quality'], test_size=0.3, random_state=40)

        
        k = 4
        num_val_samples = len(train) // k
        num_epochs = 100
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
        
        plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
        # plt.scatter(data['dense'], )
        plt.xlabel('Epochs')
        plt.ylabel('Validation MAE')
        plt.show()