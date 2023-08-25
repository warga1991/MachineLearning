import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

path = os.path.dirname(__file__)
path += '\\'

training_x = []
training_y = []

validation_x = []
validation_y = []

test_x = []
test_y = []

for data_set_index, data_purpose in enumerate(['training_data.txt', 'validation_data.txt', 'test_data.txt']):
    temp_x = []
    temp_y = []

    with open(path + data_purpose, 'r') as f:
        contents = f.read().splitlines()
    for _, line in enumerate(contents):
        line = [float(data) for data in line.split()]
        temp_x.append(line[0])
        temp_y.append(line[1])

    if data_set_index == 0:
        training_x = temp_x
        training_y = temp_y
    elif data_set_index == 1:
        validation_x = temp_x
        validation_y = temp_y
    else:
        test_x = temp_x
        test_y = temp_y

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

sine_wave_regression_model = model.fit(training_x, training_y, epochs=100, batch_size=10, validation_data=(validation_x, validation_y), callbacks=[early_stopping])

val_loss = sine_wave_regression_model.history['val_loss']
print("Validation loss:", val_loss[-1])

test_loss = model.evaluate(test_x, test_y)
print("Test loss:", test_loss)

test_predict_x = [0.8987478918617728, 0.9251636496933648, 0.8463435694934305]
predicted_y = model.predict(test_predict_x)  # Predicted target values

print("Estimated y for the test dataset:")
for i in range(len(test_predict_x)):
    print("x =", test_predict_x[i], "  Estimated y =", predicted_y[i][0])

plt.plot(sine_wave_regression_model.history['loss'], label='loss')
plt.plot(sine_wave_regression_model.history['val_loss'], label='val_loss')
plt.title('Loss Function')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()