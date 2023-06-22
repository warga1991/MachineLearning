import tensorflow as tf
from tensorflow import keras

is_working_computer = True #which computer is in use

if is_working_computer:
    path = 'C:\\Users\\u29j72\\Desktop\\private\\02_szakma\\01_programozas\\python\\01_random_things\\01_sine_things\\'
else:
    path = 'C:\\Users\\Donát\\Documents\\GitHub\\MachineLearning\\01_project_sine\\01_sine_things\\'

training_data_x = []
training_data_y = []

validation_data_x = []
validation_data_y = []

test_data_x = []
test_data_y = []

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
        training_data_x = temp_x
        training_data_y = temp_y
    elif data_set_index == 1:
        validation_data_x = temp_x
        validation_data_y = temp_y
    else:
        test_data_x = temp_x
        test_data_y = temp_y

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

regression_model = model.fit(training_data_x, training_data_y, epochs=100, validation_data=(validation_data_x, validation_data_y))

val_loss = regression_model.history['val_loss']
print("Validation loss:", val_loss[-1])

test_loss = model.evaluate(test_data_x, test_data_y)
print("Test loss:", test_loss)

test_predict_x = [0.8987478918617728, 0.9251636496933648, 0.8463435694934305]
predicted_y = model.predict(test_predict_x)  # Predicted target values

print("Estimated y for the test dataset:")
for i in range(len(test_predict_x)):
    print("x =", test_predict_x[i], "  Estimated y =", predicted_y[i][0])