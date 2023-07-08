from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

"""pixels = x_train[0].reshape((28, 28))
plt.imshow(pixels, cmap='binary')
plt.show()"""

x_train = x_train.reshape((-1, 28 * 28)) / 255.0
x_test = x_test.reshape((-1, 28 * 28)) / 255.0

assert x_train.shape == (60000, 784)
assert x_test.shape == (10000, 784)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

model = keras.Sequential([
    keras.layers.Dense(units=784, activation='relu'),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mnist_classification = model.fit(x_train, y_train, epochs= 5)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)