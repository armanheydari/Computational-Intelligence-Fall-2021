# Q2_graded
def sigmoid(Z):
  return 1.0/(1.0+np.exp(-Z))

def mean_square_error(Y, y_predict):
  temp = np.power((y_predict - Y), 2) * (1/2)
  return np.mean(temp)

def relu(Z):
  Z[Z<0] = 0
  return Z

# Q2_graded
def initialize_parameters(input_features, hidden_layers, output_features=1):
  layers = [input_features]
  layers = layers + hidden_layers
  layers.append(output_features)
  w = {}
  b = {}
  for i in range(1, len(layers)):
    w[i] = np.random.rand(layers[i], layers[i-1])
    b[i] = np.zeros((layers[i], 1))
  return w, b

# Q2_graded
def forward_propagation(X, w, b):
  Z = {0:X}
  last_layer = len(w)
  for layer in w:
    Z[layer] = np.dot(w[layer], Z[layer-1]) + b[layer]
    if layer == last_layer:
      Z[layer] = sigmoid(Z[layer])
    else:
      Z[layer] = relu(Z[layer])
  return Z

# Q2_graded
def backward_propagation(Y, Z, w, X):
  m = X.shape[1]
  last_layer = len(Z)-1
  temp = Z[last_layer]-Y
  dw = {}
  db = {}
  for layer in range(last_layer, 0, -1):
    dw[layer] = 1/m * np.dot(temp, Z[layer-1].T)
    db[layer] = 1/m * np.sum(temp, axis=1, keepdims=True)
    temp = np.dot(w[layer].T, temp)
  return dw, db

# Q2_graded
def train_mlp(X, Y, learning_rate, epochs, is_stochastic, batch_size, hidden_layers):
  n, m = X.shape
  w, b = initialize_parameters(n, hidden_layers)
  for i in range (1, epochs+1):
    if is_stochastic:
      indexes = np.random.randint(0, m, batch_size)
      x = X[:, indexes]
      y = Y[:, indexes]
    else:
      x = X
      y = Y
    Z = forward_propagation(x, w, b)
    if i%100 == 0:
      print("loss in epoch", i, ":", mean_square_error(y, Z[len(Z)-1]))
    dw, db = backward_propagation(y, Z, w, x)
    for layer in w: 
      w[layer] = w[layer] - learning_rate*dw[layer]
      b[layer] = b[layer] - learning_rate*db[layer]
  return w, b

# Q2_graded
def evaluate_model(X, w, b, threshold):
  Z = forward_propagation(X, w, b)
  y_predict = Z[len(Z)-1]
  y_predict[y_predict>threshold] = 1
  y_predict[y_predict<=threshold]= 0
  return y_predict

# Q2_graded
hidden_layers = [20,5,2,2]
learning_rate = 0.01
epochs = 2000
is_stochastic = False
batch_size = 32
threshold = 0.5

# Q2_graded
w, b = train_mlp(X, Y, learning_rate, epochs, is_stochastic, batch_size, hidden_layers)
y_predict = evaluate_model(X, w, b, threshold)
print("train accuracy:", np.sum(Y == y_predict)/Y.shape[1] * 100,"%")

