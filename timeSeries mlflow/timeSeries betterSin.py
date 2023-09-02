from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pylab as plt

# Create dataset
x = np.arange(0, np.pi * 2, 0.1)
y = np.sin(x)

# Some parameters
ACTIVE_FUN = 'tanh'
BATCH_SIZE = 1
VERBOSE=1

# Create the model
model = Sequential()
model.add(Dense(40, input_shape=(1,), activation=ACTIVE_FUN))
model.add(Dense(20, activation=ACTIVE_FUN))
model.add(Dense(64, activation=ACTIVE_FUN))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])

# Fit the model
model.fit(x, y, epochs=1500, batch_size=BATCH_SIZE, verbose=VERBOSE)

# Evaluate the model
scores = model.evaluate(x, y, verbose=VERBOSE)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

# Make predictions
xnext = np.arange(np.pi * 2, np.pi * 4, 0.1)
ynext = np.sin(x)
y_pred = model.predict(xnext)
y_predOnTrain = model.predict(x)
# Plot
plt.plot(x, y, color='blue', linewidth=1, markersize='1', label = "training data")
plt.plot(xnext, y_pred, color='green', linewidth=1, markersize='1', label = "unseen x + predictedY on x")
plt.plot(xnext, ynext, 'm.', linewidth=1, markersize='1', label = "unseen x, y")
plt.plot(x, y_predOnTrain, color='yellow', linewidth=1, markersize='1', label = "x, predictedY on x")
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()