# Q3_graded
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# Q3_graded
def render_images(images):
  columns = np.sqrt(len(images))
  rows = len(images)/columns
  fig = plt.figure(figsize=(rows*5, columns*5))
  for i in range(1, len(images)+1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(images[i-1], cmap='gray', vmin=0, vmax=255)
  plt.show()

# Q3_graded
train_path = 'data/sample/'
train_directories = sorted(os.listdir(train_path))
train_data = []
for path in train_directories:
  print(path)
  train_data.append(cv2.imread(train_path + path, cv2.IMREAD_GRAYSCALE))
train_data = np.array(train_data)
print("train data:", train_data.shape)

# Q3_graded
test_path = 'data/test/'
test_data = []
test_directories = sorted(os.listdir(test_path))
for i in range(len(test_directories)-1):
  if not(test_directories[i].endswith('.png') or test_directories[i].endswith('.jpg')):
    test_directories.pop(i)
n_train, width, height = train_data.shape
for path in test_directories:
  print(path)
  img = cv2.imread(test_path + path, cv2.IMREAD_GRAYSCALE)
  # x, y = img.shape
  # if x<width:
  #   temp = 255 * np.ones((width - x, y))
  #   img = np.concatenate((img, temp), axis = 0)
  # if y<height:
  #   temp = 255 * np.ones((width, height-y))
  #   img = np.concatenate((img, temp), axis = 1)
  temp = np.resize(img, (width, height))
  test_data.append(temp)
test_data = np.array(test_data)
n_test = test_data.shape[0]
print("test data", test_data.shape)

# Q3_graded
def find_weights(img, w):
  pattern = np.ones((width, height))
  pattern[img<200] = -1
  pattern = np.resize(pattern, (1, width*height))
  w = w + np.dot(pattern.T, pattern)
  np.fill_diagonal(w, 0.0)
  return w

# Q3_graded
def update_input(input, w):
  temp = np.ones((width, height))
  temp[input<200] = -1
  input = np.resize(temp, (width*height,1))
  for i in range(100):
    next_state = np.dot(w, input)
    input = np.sign(next_state)
  result = np.reshape(input, (width, height))*255
  result[result<0] = 0
  return result

# Q3_graded
p_train = []
for img in train_data:
  temp = np.ones((width, height))*255
  temp[img<200] = 0
  p_train.append(temp)
print("inputted patterns:")
render_images(p_train)

# Q3_graded
results = []
p_test = []
for i in range(4):
  w = find_weights(train_data[i], np.zeros((width*height, width*height)))
  temp = test_data[i]
  temp[temp<=200] = 0
  temp[temp>200] = 255
  p_test.append(temp)
  results.append(update_input(test_data[i], w))
print("images before updating with network:")
render_images(p_test)
print("images after that:")
render_images(results)

# Q3_graded
for i in range(4):
  for j in range(4):
    print("similarity of updated", test_directories[i], 
          "with", train_directories[j], "is",100*(1 - np.mean(results[i] != p_train[j])), "%")
  print("----------------------------------")

