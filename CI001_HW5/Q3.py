# Q3_graded
import random

def generate_random_binary(n):
  result=''
  for i in range(n):
    result+=str(random.randint(0, 1))
  return result


def two_point_crossover(b1, b2):
  l = len(b1)
  p1 = random.randint(0, l)
  p2 = random.randint(0, l)
  if p2<p1:
    p1, p2 = p2,p1
  b3 = b1[0:p1] + b2[p1:p2] + b1[p2:l]
  b4 = b2[0:p1] + b1[p1:p2] + b2[p2:l]
  return b3, b4


def mutation(b, probability):
  if random.random() < probability:
    m1 = random.randint(0, len(b)-1)
    if b[m1] == '0':
      b = b[0:m1] + '1' + b[m1+1:]
    else:
      b = b[0:m1] + '0' + b[m1+1:]
  return b

# Q3_graded
def fitness(x, weights, values, capacity=25):
  weight_sum = 0
  value_sum = 0
  for i in range(len(x)):
    if x[i] == '1':
      weight_sum += weights[i]
      value_sum += values[i]
  if weight_sum>25 or value_sum==0:
    return 1
  return 1.0/value_sum

# Q3_graded
def random_initialization(values, weights, number_of_initialize_population, binary_length):
  population = []
  fitnesses = []
  for i in range(number_of_initialize_population):
    x = generate_random_binary(binary_length)
    population.append(x)
    fitnesses.append(fitness(x, weights, values))
  return population, fitnesses

# Q3_graded
def genetic_knapsack(values, weights, number_of_initialize_population, binary_length, max_population, mutation_probability, generations):
  population, fitnesses = random_initialization(values, weights, number_of_initialize_population, binary_length)
  for i in range(generations):
    l = len(population)
    if l>max_population:
      population = population[l-max_population:]
      fitnesses = fitnesses[l-max_population:]

    temp = sorted(set(fitnesses))
    x_best = population[fitnesses.index(temp[0])]
    x_second_best = population[fitnesses.index(temp[1])]

    x_new_1, x_new_2 = two_point_crossover(x_best, x_second_best)

    x_new_1 = mutation(x_new_1, mutation_probability)
    x_new_2 = mutation(x_new_2, mutation_probability)
    
    if x_new_1 not in population:
      population.append(x_new_1)
      fitnesses.append(fitness(x_new_1, weights, values))
    if x_new_2 not in population:
      population.append(x_new_2)
      fitnesses.append(fitness(x_new_2, weights, values))
  return x_best

# Q3_graded
VALUES = [30, 10, 20, 50, 70, 15, 40, 25]
WEIGHTS = [2, 4, 1, 3, 5, 1, 7, 4]
NAMES = ["zomorod", "noqre", "yaqut", "almas", "berellian", "firuze", "aqiq", "kahroba"]

best_result = genetic_knapsack(VALUES,
                                WEIGHTS,
                                number_of_initialize_population=5,
                                binary_length=8,
                                max_population=10,
                                mutation_probability=0.1,
                                generations=1000
                               )

print("best chromosome:", best_result, " \n ------------------")
for i in range(len(best_result)):
   if best_result[i]=='1':
     print(NAMES[i])

