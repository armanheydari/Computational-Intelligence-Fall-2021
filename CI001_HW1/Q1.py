# Q1_graded
def initialize_parameters(n):
  w = np.random.rand(n, 1)
  b = 0.0
  return w, b

# Q1_graded
def forward_propagation(X, w, b):
  Z = np.dot(w.T, X) + b
  return Z

# Q1_graded
def sigmoid(Z):
  return 1.0/(1.0+np.exp(-Z))

# Q1_graded
def mean_square_error(Y, y_predict):
  temp = np.power((y_predict - Y), 2) * (1/2)
  return np.mean(temp)

# Q1_graded
def backward_propagation(Y, y_predict, X):
  m = X.shape[1]
  dA = y_predict-Y
  dZ = (1-y_predict)*y_predict
  dw = 1/m * np.dot(X, (dZ*dA).T)
  db = 1/m * np.sum(dZ*dA)
  return dw, db

# Q1_graded
def train_perceptron(X, Y, learning_rate, epochs, is_stochastic, batch_size):
  n, m = X.shape
  w, b = initialize_parameters(n)
  for i in range (1, epochs+1):
    if is_stochastic:
      indexes = np.random.randint(0, m, batch_size)
      x = X[:, indexes]
      y = Y[:, indexes]
    else:
      x = X
      y = Y
    Z = forward_propagation(x, w, b)
    y_predict = sigmoid(Z)
    if i%100 == 0:
      print("loss in epoch", i, ":", mean_square_error(y, y_predict))
    dw, db = backward_propagation(y, y_predict, x)
    w = w - learning_rate*dw
    b = b - learning_rate*db
  return w, b

# Q1_graded
def evaluate_model(x, w, b, threshhold):
  Z = forward_propagation(x, w, b)
  y_predict = sigmoid(Z)
  y_predict[y_predict > threshold] = 1
  y_predict[y_predict <= threshold] = 0
  return y_predict

# Q1_graded
learning_rate = 0.1
epochs = 1000
is_stochastic = True
batch_size = 16
threshold = 0.5

# Q1_graded
w, b = train_perceptron(X, Y, learning_rate, epochs, is_stochastic, batch_size)
y_predict = evaluate_model(X, w, b, threshold)
print("train accuracy:", np.sum(Y == y_predict)/Y.shape[1] * 100,"%")

