import numpy as np
import matplotlib.pyplot as plt


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def activate(x):
    return tanh(x)


def activate_derivative(x):
    return tanh_derivative(x)


file_output_weights = "output_weights.txt"
file_hidden_weights = "hidden_weights.txt"
file_hidden_bias = "hidden_bias.txt"

file_output_weights_canonical = "output_weights_canonical.txt"
file_hidden_weights_canonical = "hidden_weights_canonical.txt"
file_hidden_bias_canonical = "hidden_bias_canonical.txt"


class CalculatorNeuralNetwork:
    def __init__(self, hidden_neurons=5):
        np.random.seed(1)
        self.hidden_weights = (2 * np.random.random((2, hidden_neurons)) - 1) / 10
        self.hidden_bias = (2 * np.random.random((1, hidden_neurons)) - 1) / 10
        self.output_weights = (2 * np.random.random((hidden_neurons, 4)) - 1) / 10

    def set_weights(self):
        self.hidden_weights = np.loadtxt(file_hidden_weights)
        self.hidden_bias = np.loadtxt(file_hidden_bias)
        self.output_weights = np.loadtxt(file_output_weights)

    def set_weights_canonical(self):
        self.hidden_weights = np.loadtxt(file_hidden_weights_canonical)
        self.hidden_bias = np.loadtxt(file_hidden_bias_canonical)
        self.output_weights = np.loadtxt(file_output_weights_canonical)

    def save_weights(self):
        np.savetxt(file_output_weights, self.output_weights)
        np.savetxt(file_hidden_weights, self.hidden_weights)
        np.savetxt(file_hidden_bias, self.hidden_bias)

    def save_weights_canonical(self):
        np.savetxt(file_output_weights_canonical, self.output_weights)
        np.savetxt(file_hidden_weights_canonical, self.hidden_weights)
        np.savetxt(file_hidden_bias_canonical, self.hidden_bias)

    def backpropagation(self, inputs, output_delta, hidden_layer_output, learning_rate=0.1):
        hidden_error = np.dot(output_delta, self.output_weights.T)
        hidden_delta = hidden_error * activate_derivative(hidden_layer_output)

        self.output_weights += learning_rate * np.dot(hidden_layer_output.T, output_delta.reshape(1, -1))
        self.hidden_weights += learning_rate * np.dot(inputs.T, hidden_delta)
        self.hidden_bias += learning_rate * 2 * hidden_delta

    def predict(self, x1, x2):
        inputs = np.array([x1, x2]).reshape(1, -1)
        hidden_layer_input = np.dot(inputs, self.hidden_weights) + self.hidden_bias
        hidden_layer_output = activate(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.output_weights)
        output_layer_output = output_layer_input
        return output_layer_output.reshape(-1)

    def train(self, X_train, y_train, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            total_error = 0

            for i in range(len(X_train)):
                inputs = X_train[i].reshape(1, -1)
                hidden_layer_input = np.dot(inputs, self.hidden_weights) + self.hidden_bias
                hidden_layer_output = activate(hidden_layer_input)
                output_layer_input = np.dot(hidden_layer_output, self.output_weights)
                output_layer_output = output_layer_input

                error = y_train[i].reshape(-1) - output_layer_output.reshape(-1)
                total_error += np.abs(error)

                output_delta = error
                self.backpropagation(inputs, output_delta, hidden_layer_output, learning_rate)

            mean_absolute_error = total_error / len(X_train)
            print(f"Epoch {epoch + 1}/{epochs}, Mean Absolute Error: {mean_absolute_error}")
        self.save_weights()
        # self.save_weights_canonical()


X_train = np.random.rand(10000, 2).round(6)

for i in X_train:
    if i[0] < 0.01:
        i[0] = 0.01
    if i[1] < 0.01:
        i[1] = 0.01

y_train = np.zeros((10000, 4))
for i, (x1, x2) in enumerate(X_train):
    y_train[i, 0] = x1 + x2
    y_train[i, 1] = x1 - x2
    y_train[i, 2] = np.log(x1) + np.log(x2)
    y_train[i, 3] = np.log(x1) - np.log(x2)

X_train_canonical = np.random.rand(1000, 2).round(4) * 0.4 + 0.1

y_train_canonical = np.zeros((1000, 4))
for i, (x1, x2) in enumerate(X_train_canonical):
    y_train_canonical[i, 0] = x1 + x2
    y_train_canonical[i, 1] = x1 - x2
    y_train_canonical[i, 2] = x1 * x2
    y_train_canonical[i, 3] = x1 / x2

nn = CalculatorNeuralNetwork(hidden_neurons=2000)

#nn.train(X_train, y_train, epochs=10000, learning_rate=0.01)
nn.set_weights()
# nn.set_weights_canonical()

'''x = np.linspace(1, 1000, 100)
y = np.linspace(1, 1000, 100)
X, Y = np.meshgrid(x, y)'''


def z(X, Y, net):
    Z1 = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            Z1[i, j] = np.abs(net.predict(X[i, j] / 1000, Y[i, j] / 1000)[0] * 1000 - abs(X[i, j] + Y[i, j])) / abs(X[i, j] + Y[i, j])

    Z2 = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            if abs(X[i, j] - Y[i, j]) == 0:
                Z2[i, j] = 0
            else:
                Z2[i, j] = np.abs(np.abs(net.predict(X[i, j] / 1000, Y[i, j] / 1000)[1]) * 1000 - abs(X[i, j] - Y[i, j])) / abs(X[i, j] - Y[i, j])

    Z3 = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            Z3[i, j] = np.abs(np.exp(net.predict(X[i, j] / 1000, Y[i, j] / 1000)[2]) * 1000000 - abs(X[i, j] * Y[i, j])) / abs(X[i, j] * Y[i, j])

    Z4 = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            Z4[i, j] = np.abs(np.exp(net.predict(X[i, j] / 1000, Y[i, j] / 1000)[3]) - abs(X[i, j] / Y[i, j])) / abs(X[i, j] / Y[i, j])
    return [Z1, Z2, Z3, Z4]


def z_canonical(X, Y, net):
    Z1 = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            Z1[i, j] = np.abs(net.predict(X[i, j] / 1000, Y[i, j] / 1000)[0] * 1000 - abs(X[i, j] + Y[i, j])) / abs(X[i, j] + Y[i, j])

    Z2 = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            if abs(X[i, j] - Y[i, j]) == 0:
                Z2[i, j] = 0
            else:
                Z2[i, j] = np.abs(np.abs(net.predict(X[i, j] / 1000, Y[i, j] / 1000)[1]) * 1000 - abs(X[i, j] - Y[i, j])) / abs(X[i, j] - Y[i, j])

    Z3 = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            Z3[i, j] = np.abs(net.predict(X[i, j] / 1000, Y[i, j] / 1000)[2] * 1000000 - abs(X[i, j] * Y[i, j])) / abs(X[i, j] * Y[i, j])

    Z4 = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            Z4[i, j] = np.abs(net.predict(X[i, j] / 1000, Y[i, j] / 1000)[3] - abs(X[i, j] / Y[i, j])) / abs(X[i, j] / Y[i, j])
    return [Z1, Z2, Z3, Z4]


def results():
    net_good = CalculatorNeuralNetwork(hidden_neurons=2000)
    net_good.set_weights()

    net_canonical = CalculatorNeuralNetwork(hidden_neurons=1000)
    net_canonical.set_weights_canonical()

    x = np.linspace(50, 980, 100)
    y = np.linspace(50, 980, 100)
    X, Y = np.meshgrid(x, y)

    good_results = z(X, Y, net_good)
    canonical_results = z_canonical(X, Y, net_canonical)

    fig = plt.figure()
    ax1 = []
    ax2 = []
    actions = ["+", "-", "*", "/"]

    for i in range(4):
        ax1.append(fig.add_subplot(241 + i, projection='3d'))
        ax1[i].plot_surface(X, Y, good_results[i], cmap='viridis')
        ax1[i].set_xlabel('X')
        ax1[i].set_ylabel('Y')
        ax1[i].set_zlabel('Error')
        ax1[i].set_title(f'Set 1 {actions[i]}')
    ax1[0].set_zlim(0, 1)
    ax1[1].set_zlim(0, 1)
    ax1[2].set_zlim(0, 1)
    ax1[3].set_zlim(0, 1)

    for i in range(4):
        ax2.append(fig.add_subplot(245 + i, projection='3d'))
        ax2[i].plot_surface(X, Y, canonical_results[i], cmap='viridis')
        ax2[i].set_xlabel('X')
        ax2[i].set_ylabel('Y')
        ax2[i].set_zlabel('Error')
        ax2[i].set_title(f'Set 2 {actions[i]}')
    ax2[0].set_zlim(0, 1)
    ax2[1].set_zlim(0, 1)
    ax2[2].set_zlim(0, 1)
    ax2[3].set_zlim(0, 1)

    plt.subplots_adjust(wspace=1, hspace=0.4)
    plt.show()


results()

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Prediction')
ax.set_title('Neural Network Prediction Surface')

plt.show()'''


while True:
    x1, x2 = list(map(int, input().split()))
    result = nn.predict(x1 / 1000, x2 / 1000)
    print(f"Input: ({x1}, {x2}), Predicted Result: {[result[0] * 1000, result[1] * 1000, np.exp(result[2]) * 1000000, np.exp(result[3])]}")
    print(f"Errors: {[np.abs(result[0] * 1000 - (x1 + x2)),np.abs(np.abs(result[1] * 1000) - np.abs(x1 - x2)), np.abs(np.exp(result[2]) * 1000000 - x1 * x2), np.abs(np.exp(result[3]) - x1 / x2)]}")

