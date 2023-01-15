# Q4_graded
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin
from functools import reduce

!wget -N -q 'https://github.com/armanheydari/my-datas/raw/master/data.csv'

# Q4_graded
data = pd.read_csv('data.csv', header=None)
data = data.to_numpy()
data_no = data.shape[0]
cities = np.zeros((data_no, 2))
for i in range(data_no):
  [index, x, y] = data[i, 0].split(' ')
  cities[i, 0] = float(x)
  cities[i, 1] = float(y)

# Q4_graded
def plot_solution(cities, sol):
    plt.figure('Traveling Salesman Problem')
    plt.plot(cities[:, 0], cities[:, 1], 'b*')
    plt.plot(sol[:, 0], sol[:, 1], '-rx')
    plt.title('Traveling Salesman Problem')
    plt.grid()
    plt.show()

# Q4_graded
class KohonenNN(object):
    def __init__(self, N, r=None):
        self.N = N
        if r is not None:
            self.r = r * np.random.rand()
            self.w = self.r * np.array(list(map(lambda theta: (cos(2 * pi * theta) + 0.5, sin(2 * pi * theta) + 0.5),
                                                np.linspace(0, 2 * pi, N))))
        else:
            self.r = np.random.rand()  # I assume r == 1.
            self.w = self.r * np.array(list(map(lambda theta: (cos(2 * pi * theta) + 0.5, sin(2 * pi * theta) + 0.5),
                                                np.linspace(0, 2 * pi, N))))

    def optimize(self, C, d, eta, Nk, sigma):
        for k in range(Nk):
          sigma -= (0.3 * self.N - 0.001 * self.N) / Nk
          for j in np.random.permutation(int(self.N / 2)):
              r_min = np.argmin(np.linalg.norm(self.w - C[j], axis=1))
              for i in range(self.N):
                  self.w[i] += eta * d(i, r_min, sigma) * (C[j] - self.w[i])
        return self.w

# Q4_graded
N = cities.shape[0] * 2
eta = 0.2
Nk = 2000
sigma = 0.3 * N
dist = lambda r, r_min, s: np.exp(-(np.abs(r - r_min)**2) / (2 * s**2)) # neighborhood function

KohNN = KohonenNN(N, r=1.0)
for i in range(1, 6):
  KohNN.optimize(cities, dist, eta, int(Nk/5), sigma)
  sol = np.vstack((KohNN.w, KohNN.w[0, :]))
  print("iteration", i*int(Nk/5), "done!")
  plot_solution(cities, sol)
  print()

