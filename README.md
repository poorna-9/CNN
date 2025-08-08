# 🧠 NeuroLite-CNN

A modular, NumPy-only Convolutional Neural Network (CNN) framework built completely from scratch. Inspired by TensorFlow/Keras, but designed for full transparency and educational flexibility.

## 🚀 Features

- Build CNN architectures using simple class-based APIs.
- Supports:
  - **Convolutional layers**
  - **MaxPooling layers**
  - **Fully connected layers (Dense)**
  - Common **activations** (ReLU, Sigmoid, Tanh, Softmax)
  - **Cross-Entropy**, **MSE** losses
  - **Backpropagation through Conv and Pooling layers**
- Trainable on **MNIST** and other small image datasets.
- No external deep learning libraries used — **only NumPy**.
- Designed for clarity, learning, and customization.

## 🧩 Example Usage

```python
from neuro_lite_cnn import Conv2D, MaxPool2D, Flatten, Dense, ReLU, Softmax, CrossEntropyLoss, NeuralNetwork

model = NeuralNetwork()
model.add(Conv2D(input_shape=(1, 28, 28), num_filters=8, kernel_size=3))
model.add(ReLU())
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(128))
model.add(ReLU())
model.add(Dense(10))
model.add(Softmax())

model.compile(loss=CrossEntropyLoss(), learning_rate=0.01)
model.fit(X_train, y_train, epochs=10, batch_size=32)
model.evaluate(X_test, y_test)
