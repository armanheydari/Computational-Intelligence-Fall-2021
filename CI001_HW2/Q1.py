# Q1_graded
import cv2
!wget -N -q 'https://github.com/armanheydari/my-datas/raw/master/input.jpg'
input_image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
print("input numpy array's shape:", input_image.shape)

# Q1_graded
import numpy as np
def initialize_map(weight_num):
  w = np.random.uniform(0, input_image.shape[0], (weight_num, 2))
  return w.astype(int)

# Q1_graded
from random import seed, randint
seed(1)
def choose_sample(input):
  x_index = randint(0, input.shape[0]-1)
  y_index = randint(0, input.shape[1]-1)
  sample = input[x_index, y_index]
  index = np.array([x_index, y_index])
  return sample, index

# Q1_graded
def euclidean_distance(x1, x2):
  return np.sqrt(np.sum(np.power(x1-x2, 2)))

# Q1_graded
def find_closest(sample, w):
  min_distance = float('inf')
  for weight in w:
    d = euclidean_distance(weight, sample)
    if min_distance>d:
      min_distance = d
      closest = weight
  return closest

# Q1_graded
def update_weights(sample, index, closest, w):
  for i in range(w.shape[0]):
    interval = euclidean_distance(w[i], closest)
    if interval<radius and sample>0:
      h = np.exp(-interval ** 2 / (2 * (radius ** 2)))
      w[i] = w[i] + sample * h * (index-w[i])
  return w

# Q1_graded
def weights_to_image(w):
  image = 255*np.ones(input_image.shape)
  for [x, y] in w:
    image[x,y] = 0
  return image

# Q1_graded
from matplotlib import pyplot as plt
def render_images(images):
  columns = np.sqrt(len(images))
  rows = len(images)/columns
  fig = plt.figure(figsize=(rows*5, columns*5))
  for i in range(1, len(images)+1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(images[i-1], cmap='gray', vmin=0, vmax=255)
  plt.show()

# Q1_graded
epochs = 20000
weight_num = 22*64
radius = 2.5

# Q1_graded
images = []
temp = 1-(input_image/255.0)
w = initialize_map(weight_num)
for epoch in range(1, epochs+1):
  sample, index = choose_sample(temp)
  closest = find_closest(index, w)
  w = update_weights(sample, index, closest, w)
  if epoch%1000 == 0:
    print("epoch", epoch, "done!")
  if epoch in [1, epochs/100, epochs/50, epochs/20, epochs/10, epochs/5, epochs/4, epochs/2, epochs]:
    w[w>input_image.shape[0]-1] = input_image.shape[0]-1
    w[w<0] = 0
    images.append(weights_to_image(w))
render_images(images)

