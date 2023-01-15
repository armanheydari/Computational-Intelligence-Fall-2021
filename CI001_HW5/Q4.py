# Q4_graded
import numpy as np
def random_initialization(particles_no, mlp_weights_no):
  particles = np.random.rand(particles_no, mlp_weights_no)
  velocities = np.random.rand(particles_no, mlp_weights_no)
  pbest = []
  gbest = (particles[0], float('inf'))
  for p in particles:
    pbest.append((p, float('inf')))
  return particles, velocities, pbest, gbest

# Q4_graded
def mean_squared_error(y_predict, y):
  return np.mean((y_predict - y) ** 2)

# Q4_graded
def relu(x):
  x[x<0] = 0
  return x

# Q4_graded
def particle_to_layer(particle, units=6):
  input_length = int(particle.shape[0]/units) - 1
  hidden_layer = np.zeros((units, input_length))
  output_layer = np.zeros((units, 1))
  for i in range(units):
    for j in range(input_length):
      hidden_layer[i, j] = particle[i*input_length+j]
  
  for i in range(units):
    output_layer[i] = particle[units*input_length+i]
  
  return hidden_layer, output_layer

# Q4_graded
def predict(particle, x_train):
  hidden_layer, output_layer = particle_to_layer(particle)
  z = np.dot(x_train, hidden_layer.T)
  y_predict = np.dot(z, output_layer)
  return y_predict

# Q4_graded
def update_bests(particles, fitnesses, pbest, gbest):
  for i in range(len(fitnesses)):
    p, f = pbest[i]
    if f>=fitnesses[i]:
      pbest[i] = (particles[i], fitnesses[i])

    if gbest[1]>=fitnesses[i]:
      gbest = (particles[i], fitnesses[i])
  return pbest, gbest

# Q4_graded
from random import random
def update_velocities(particles, velocities, w, c1, c2, pbest, gbest):
  new_velocities = np.zeros(velocities.shape)
  for i in range(velocities.shape[0]):
    delta_p = pbest[i][0] - particles[i]
    delta_g = gbest[0] - particles[i]
    new_velocities[i] = w*velocities[i] + c1*random()*delta_p + c2*random()*delta_g
  return new_velocities

# Q4_graded
def update_particles(particles, new_velocities):
  new_particles = particles + new_velocities
  return new_particles

# Q4_graded
def train_mlp(x_train, y_train, epochs, w, c1, c2, particles_no, mlp_weights_no):
  particles, velocities, pbest, gbest = random_initialization(particles_no, mlp_weights_no)
  for epoch in range(epochs):
    fitnesses = []
    for particle in particles:
      y_predict = predict(particle, x_train)
      fitnesses.append(mean_squared_error(y_predict, y_train))
    pbest, gbest = update_bests(particles, fitnesses, pbest, gbest)
    print('best fitness on epoch', epoch+1, 'is:', int(gbest[1]))
    velocities = update_velocities(particles, velocities, w, c1, c2, pbest, gbest)
    particles = update_particles(particles, velocities)
  return gbest[0]

# Q4_graded
x_train = np.random.randint(-10000, 10000, (10000, 1))
y_train = x_train
x_test = np.random.randint(-10000, 10000, (100, 1))
y_test = x_test
best_particle = train_mlp(x_train,
                          y_train,
                          epochs = 20,
                          w = 0.5,
                          c1 = 3,
                          c2 = 1,
                          particles_no = 200,
                          mlp_weights_no= 12
                )
print('------------------------------------------------------')
y_predict_train = np.rint(predict(best_particle, x_train))
print("accuracy on train:", 100 - 100 * (np.mean(y_predict_train != y_train)), '%')
y_predict_test = np.rint(predict(best_particle, x_test))
print("accuracy on test:", 100 - 100 * (np.mean(y_predict_test != y_test)), '%')

# Q4_graded
x_train = np.random.randint(-10000, 10000, (10000, 2))
y_train = np.sum(x_train, axis=1, keepdims=True)
x_test = np.random.randint(-10000, 10000, (100, 2))
y_test = np.sum(x_test, axis=1, keepdims=True)
best_particle = train_mlp(x_train,
                          y_train,
                          epochs = 50,
                          w = 0.5,
                          c1 = 3,
                          c2 = 1,
                          particles_no = 500,
                          mlp_weights_no= 18
                )
print('------------------------------------------------------')
y_predict_train = np.rint(predict(best_particle, x_train))
print("accuracy on train:", 100 - 100 * (np.mean(y_predict_train != y_train)), '%')
y_predict_test = np.rint(predict(best_particle, x_test))
print("accuracy on test:", 100 - 100 * (np.mean(y_predict_test != y_test)), '%')

# Q4_graded
from math import sin, cos

x_train = np.random.randint(-10000, 10000, (10000, 2))
y_train = np.zeros((10000, 1))
for i in range(x_train.shape[0]):
  y_train[i] = sin(x_train[i,0]) + cos(x_train[i,1])

x_test = np.random.randint(-10000, 10000, (100, 2))
y_test = np.zeros((100, 1))
for i in range(x_test.shape[0]):
  y_test[i] = sin(x_test[i,0]) + cos(x_test[i,1])

best_particle = train_mlp(x_train,
                          y_train,
                          epochs = 100,
                          w = 0.5,
                          c1 = 3,
                          c2 = 1,
                          particles_no = 200,
                          mlp_weights_no= 18
                )

# Q4_graded
print("the numbers are float here so we can't take accuracy as a good method because the network can't predict the exact value!")
y_predict_train = predict(best_particle, x_train)
print("mean square error on train:", mean_squared_error(y_predict_train, y_train))
y_predict_test = np.rint(predict(best_particle, x_test))
print("mean square error on test:", mean_squared_error(y_predict_test, y_test))

# Q4_graded
x_train = np.random.randint(-10, 10, (100, 2))
y_train = np.prod(x_train, axis=1, keepdims=True)

x_test = np.random.randint(-10, 10, (10, 2))
y_test = np.prod(x_test, axis=1, keepdims=True)

best_particle = train_mlp(x_train,
                          y_train,
                          epochs = 100,
                          w = 0.5,
                          c1 = 3,
                          c2 = 1,
                          particles_no = 100,
                          mlp_weights_no= 18
                )
print('------------------------------------------------------')
y_predict_train = np.rint(predict(best_particle, x_train))
print("accuracy on train:", 100 - 100 * (np.mean(y_predict_train != y_train)), '%')
y_predict_test = np.rint(predict(best_particle, x_test))
print("accuracy on test:", 100 - 100 * (np.mean(y_predict_test != y_test)), '%')

