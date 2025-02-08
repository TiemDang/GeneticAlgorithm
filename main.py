from ClassGA import population
from ClassNeuralNetwork import NeuralNet
import numpy as np
#import cupy as np
import matplotlib.pyplot as plt



"""
Smallest unit is gen
Unit : gen > chromosome > individual > population
6 gen = 1 chromosome
2 chromosome = 1 individual
(n) individual = 1 population [n is a number ]
"""


# Create variable
range_decode_X = [-1, 1]
range_decode_Y = [-1, 1]
population_size = 1000 # Number of population
num_chromo = 30 # Number of chromosme
num_gene = 10 # Gene on 1 chromosome


# ------------
min_list = []
max_generation = 10
min_fitness = 9999
best_solution = np.zeros(num_chromo)


# mean_fitness
mean_fitness = np.zeros(max_generation)


# X matrix
X1 = np.array([0.3, 0.35, 0.4, 0.8, 0.9, 1.0, 1.2, 1.6, 2.0 ])
X2 = np.array([0.3, 0.4, 0.5, 0.75, 0.7, 0.8, 0.4, 0.5, 0.5 ])
X = np.vstack((X1,X2)) # ----> Input matrix

# GA variable
j = np.zeros(population_size) #

# Run
Y_draw = None

initialize = population(range_decode_X, range_decode_Y, population_size, num_chromo, num_gene)
population = initialize.create_population()
for generation in range(max_generation):
    decode_population = initialize.decode_gen(population)
    for individual in range (population_size) :
        W = decode_population[individual][0:20].reshape(2, 10)
        V = decode_population[individual][20:30].reshape(10, 1)
        Neural = NeuralNet(X, W, V)
        Y_head = Neural.FeedForward()
        j[individual] = Neural.cost_function()
        if j[individual] < (min_fitness):
            min_fitness = j[individual]
            best_solution = decode_population[individual]
            Y_draw = Y_head # draw the best individual 
    
    
        #print(decode_population[0], len(decode_population[0]))
        """
        fitness_list = method.fitness(decode_population)
        for index in range(len(decode_population)):
            if (fitness_list[index] < min_fitness ):
                min_fitness = fitness_list[index]
                best_solution = decode_population[index]
        """
    
    """
    This is for test
    
    if (generation > 2995 ):
        print("List fitness : {}, len : {}".format(fitness_list, len(fitness_list)))
    
    """
    min_list.append(min_fitness)
    chosen_population = initialize.selection(j, population)
    cross_population = initialize.crossover(chosen_population)
    population = initialize.mutation(cross_population, 0.3)
    mean_fitness[generation] = np.mean(j)
    print(f'Generation: {generation}, j: {min_fitness}, j_mean: {np.mean(j)}, invidual: {best_solution}')

Y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
ypoints = np.array(min_list)
#print(mean_fitness, np.mean(mean_fitness)) # Draw mean fitness of each generation


# Visualize
plt.figure(1)
plt.plot(ypoints, color = 'Red')
plt.xlabel('Generation') 
plt.ylabel('Fitness')
plt.xlim
#------------------------------------
plt.figure(2)
plt.plot(Y_draw[0], color = 'Blue')
plt.plot(Y, color = 'Red')
plt.plot()
plt.show()




