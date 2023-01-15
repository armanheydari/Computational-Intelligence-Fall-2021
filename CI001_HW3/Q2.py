# Q2_graded
import numpy as np

patterns = np.array([[1,-1,1,1,-1,1], [1,-1,1,-1,1,-1], [1,1,1,1,1,-1]])
patterns

# Q2_graded
n = patterns.shape[1]
p = patterns.shape[0]
w = np.zeros((n, n))
for i in range(0, n):
  for j in range(0, n):
    for k in range(0, p):
      if i!=j:
        w[i,j] += patterns[k, i] * patterns[k, j]
w

# Q2_graded
sign = np.sign(np.dot(patterns, w))
if np.array_equal(sign, patterns):
  print("all the patterns are stable")
else:
  print("something is wrong!")

# Q2_graded
def update_hopfield(input):
  THRESHOLD = 0 
  flag = True
  counter = 0
  while flag:
    counter +=1
    next_state = np.dot(w, input)
    next_state[next_state>THRESHOLD] = 1
    next_state[next_state<=THRESHOLD] = -1
    if np.array_equal(input, next_state):       #updating ends when we have similar states sequentially.
      flag = False
    input = next_state
    
    if counter == 10000:
      print("this is iteration", counter, "so the hopfield neural network can't find an optimum state!")
      return -1
  return next_state

# Q2_graded
inputs = np.array([[-1,1,-1,-1,1,-1], [-1,1,-1,1,-1,1], [-1,-1,-1,-1,-1,1]])
for input in inputs:
  if np.array_equal(input, update_hopfield(input)):
    print(input, " is stable!")
  else:
    print(input, " is unstable!")

# Q2_graded
test = update_hopfield(np.array([1,1,1,1,1,1]))

