# Q2.1_graded
import numpy as np
from keras.layers import Layer
from keras import backend as K
import keras
from matplotlib import pyplot as plt

# Q2.1_graded
class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

# Q2.1_graded
x_train = np.array([[0.2, 0.4], [0.2,0.5], [0.2,0.6], [0.3,0.3], [0.4,0.3], [0.6,0.3], [0.7,0.3], [0.8, 0.4], [0.8,0.5], [0.8,0.6],
             [0.3, 0.4], [0.3,0.5], [0.3,0.6], [0.7, 0.4],[0.4,0.7], [0.5,0.7], [0.6,0.7], [0.7,0.5], [0.7,0.6],
             [0.4,0.4], [0.5,0.4], [0.6,0.4], [0.4,0.5], [0.5,0.5], [0.6,0.5], [0.4,0.6], [0.5,0.6], [0.6,0.6], [0.15, 0.25]])
y_train = np.array([2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,3])

x_temp = np.random.uniform(0.1, 0.8, 30)
y_temp = np.random.uniform(0.1, 0.9, 30)
x_test = np.zeros((30,2))
for i in range(0,30):
  x_test[i,0] = x_temp[i]
  x_test[i,1] = y_temp[i]
print("x_train's shape:", x_train.shape)
print("y_train's shape:", y_train.shape)
print("x_test's shape:", x_test.shape)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='plasma')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('train datas')
plt.show()

# Q2.1_graded
RBF_model = keras.models.Sequential(layers=[
                                            keras.layers.Input(2),
                                            RBFLayer(15, 0.5),
                                            keras.layers.Dense(4, activation='softmax'),                      
])

RBF_model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

RBF_history = RBF_model.fit(
    x_train,
    y_train,
    epochs = 20000,
    verbose = 0
)

print("RBF accuracy on train set:", RBF_model.evaluate(x_train, y_train)[1] * 100, "%")

# Q2.1_graded
plt.plot(RBF_history.history['accuracy'])
plt.title('RBF accuracy')
plt.show()
plt.plot(RBF_history.history['loss'])
plt.title('RBF loss')
plt.show()

# Q2.1_graded
y_rbf = np.argmax(RBF_model.predict(x_test), axis=1)

plt.scatter(x_test[:, 0], x_test[:, 1], c=y_rbf, cmap='plasma')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('test results with RBF')
plt.show()

