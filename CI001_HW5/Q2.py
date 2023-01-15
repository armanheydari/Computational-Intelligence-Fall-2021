# Q2_graded
from random import randint
def generate_random_binary(n):
  result=''
  for i in range(n):
    result+=str(randint(0, 1))
  return result

# Q2_graded
def binary_to_decimal(binary_number):
  c = len(binary_number) - 1
  result=0
  for digit in binary_number:
    result += 2**c * int(digit)
    c -= 1
  return result

# Q2_graded
def fitness(x, y):
  return abs(2*x*x + y - 59)

# Q2_graded
def two_point_crossover(b1, b2):
  l = len(b1)
  p1 = randint(0, l)
  p2 = randint(0, l)
  if p2<p1:
    p1, p2 = p2,p1
  b3 = b1[0:p1] + b2[p1:p2] + b1[p2:l]
  b4 = b2[0:p1] + b1[p1:p2] + b2[p2:l]
  return b3, b4

# Q2_graded
import random
def mutation(b, probability):
  if random.random() < probability:
    m1 = randint(0, 5)
    if b[m1] == '0':
      b = b[0:m1] + '1' + b[m1+1:]
    else:
      b = b[0:m1] + '0' + b[m1+1:]
  return b

# Q2_graded
def add_chromosome(x, y, population, fitnesses):
  f = fitness(binary_to_decimal(x), binary_to_decimal(y))
  if f not in fitnesses:
    population.append((x, y))
    fitnesses.append(f)
  return population, fitnesses

# Q2_graded
def random_initialization(n, binary_length):
  population = []
  fitnesses = []
  for i in range(n):
    population, fitnesses = add_chromosome(generate_random_binary(binary_length), generate_random_binary(binary_length), population, fitnesses)
  return population, fitnesses

# Q2_graded
def genetic_programming(number_of_initialize_population, binary_length, max_population, mutation_probability):
  counter = 0
  population, fitnesses = random_initialization(number_of_initialize_population, binary_length)
  while True:
    l = len(population)
    if l>max_population:
      population = population[l-max_population:]
      fitnesses = fitnesses[l-max_population:]
    
    temp = sorted(set(fitnesses))
    best_couple_index = fitnesses.index(temp[0])
    second_best_couple_index = fitnesses.index(temp[1])
    
    x_best, y_best = population[best_couple_index]
    x_second_best, y_second_best = population[second_best_couple_index]

    if temp[0] == 0:
      print("result found after", counter, "generations:")
      print('x , y =', x_best, ",", y_best)
      print("which means x=", binary_to_decimal(x_best),", y=", binary_to_decimal(y_best))
      break
    
    x_new_1, x_new_2 = two_point_crossover(x_best, x_second_best)
    y_new_1, y_new_2 = two_point_crossover(y_best, y_second_best)

    x_new_1 = mutation(x_new_1, mutation_probability)
    x_new_2 = mutation(x_new_2, mutation_probability)
    y_new_1 = mutation(y_new_1, mutation_probability)
    y_new_2 = mutation(y_new_2, mutation_probability)
    
    population, fitnesses = add_chromosome(x_new_1, y_new_1, population, fitnesses)
    population, fitnesses = add_chromosome(x_new_2, y_new_1, population, fitnesses)
    population, fitnesses = add_chromosome(x_new_1, y_new_2, population, fitnesses)
    population, fitnesses = add_chromosome(x_new_2, y_new_2, population, fitnesses)

    counter+=1

# Q2_graded
genetic_programming(number_of_initialize_population=10,
                    binary_length=6,
                    max_population=20,
                    mutation_probability=0.1)

