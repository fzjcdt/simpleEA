import numpy as np
import copy as cp
import matplotlib.pyplot as plt

LENGTH = 5
MUTATION_RATE = 1.0 / LENGTH


class Individual(object):

    def __init__(self):
        self.gene = np.random.randint(0, 2, LENGTH)
        self.fitness = self.calculate_fitness(self.gene)

    def update_fitness(self):
        self.fitness = self.calculate_fitness(self.gene)

    @staticmethod
    def calculate_fitness(gene):
        fitness = 0
        for temp in gene:
            fitness = fitness * 2 + temp

        return fitness ** 2

    def mutation(self):
        for i in range(len(self.gene)):
            if np.random.rand() < MUTATION_RATE:
                self.gene[i] = 1 - self.gene[i]

        self.update_fitness()


class Population(object):

    def __init__(self, size=5, generation_num=20):
        self.population = [Individual() for i in range(size)]
        self.so_far_max_fitness = [max([indi.fitness for indi in self.population])]
        self.max_fitness_in_each_generation = [max([indi.fitness for indi in self.population])]
        self.max_fitness_in_each_eval = [self.population[0].fitness]
        for indi in self.population[1:]:
            self.max_fitness_in_each_eval.append(max(indi.fitness, self.max_fitness_in_each_eval[-1]))

        for i in range(generation_num - 1):
            self.population = self.next_generation()
            self.so_far_max_fitness.append(
                max(max([indi.fitness for indi in self.population]), self.so_far_max_fitness[-1]))
            self.max_fitness_in_each_generation.append(max([indi.fitness for indi in self.population]))
            for indi in self.population:
                self.max_fitness_in_each_eval.append(max(indi.fitness, self.max_fitness_in_each_eval[-1]))

    def next_generation(self):
        new_population = []
        size = len(self.population)
        while(len(new_population)) < size:
            new_population.append(self.mutation())
            new_population.extend(self.crossover())

        return new_population[:size]

    def select(self):
        total_fitness, cur_fitness = 0, 0
        for individual in self.population:
            total_fitness += individual.fitness

        point = np.random.rand() * total_fitness
        for individual in self.population:
            cur_fitness += individual.fitness
            if cur_fitness >= point:
                return individual

    def mutation(self):
        individual = cp.deepcopy(self.select())
        individual.mutation()
        return individual

    def crossover(self):
        indiv_1 = cp.deepcopy(self.select())
        indiv_2 = cp.deepcopy(self.select())
        p = np.random.randint(1, LENGTH)

        indiv_1.gene[p:], indiv_2.gene[p:] = indiv_2.gene[p:], indiv_1.gene[p:]
        indiv_1.update_fitness()
        indiv_2.update_fitness()

        return indiv_1, indiv_2


def main():
    num = 50
    p1 = Population(size=4, generation_num=num)
    p1_so_far_max_fitness = np.array(p1.so_far_max_fitness)
    p1_max_fitness_in_each_generation = np.array(p1.max_fitness_in_each_generation)

    p2 = Population(size=8, generation_num=num)
    p2_so_far_max_fitness = np.array(p2.so_far_max_fitness)
    p2_max_fitness_in_each_generation = np.array(p2.max_fitness_in_each_generation)

    p3 = Population(size=15, generation_num=num)
    p3_so_far_max_fitness = np.array(p3.so_far_max_fitness)
    p3_max_fitness_in_each_generation = np.array(p3.max_fitness_in_each_generation)

    for i in range(29):
        p1 = Population(size=4, generation_num=num)
        p1_so_far_max_fitness = p1_so_far_max_fitness + np.array(p1.so_far_max_fitness)
        p1_max_fitness_in_each_generation = p1_max_fitness_in_each_generation + np.array(p1.max_fitness_in_each_generation)

        p2 = Population(size=8, generation_num=num)
        p2_so_far_max_fitness = p2_so_far_max_fitness + np.array(p2.so_far_max_fitness)
        p2_max_fitness_in_each_generation = p2_max_fitness_in_each_generation + np.array(p2.max_fitness_in_each_generation)

        p3 = Population(size=15, generation_num=num)
        p3_so_far_max_fitness = p3_so_far_max_fitness + np.array(p3.so_far_max_fitness)
        p3_max_fitness_in_each_generation = p3_max_fitness_in_each_generation + np.array(p3.max_fitness_in_each_generation)

    x = [i for i in range(num)]
    plt.plot(x, p1_so_far_max_fitness / 30, label='size=4')
    plt.plot(x, p2_so_far_max_fitness / 30, label='size=8')
    plt.plot(x, p3_so_far_max_fitness / 30, label='size=15')
    plt.legend()
    plt.title('Average performance of 30 runs')
    plt.xlabel('generation')
    plt.ylabel('best so far fitness')
    plt.show()

    plt.plot(x, p1_max_fitness_in_each_generation / 30, label='size=4')
    plt.plot(x, p2_max_fitness_in_each_generation / 30, label='size=8')
    plt.plot(x, p3_max_fitness_in_each_generation / 30, label='size=15')
    plt.legend()
    plt.title('Average performance of 30 runs')
    plt.xlabel('generation')
    plt.ylabel('best fitness in this generation')
    plt.show()


    p1 = Population(size=4, generation_num=50)
    p1_max_fitness_in_each_eval = np.array(p1.max_fitness_in_each_eval)

    p2 = Population(size=8, generation_num=25)
    p2_max_fitness_in_each_eval = np.array(p2.max_fitness_in_each_eval)

    p3 = Population(size=25, generation_num=8)
    p3_max_fitness_in_each_eval = np.array(p3.max_fitness_in_each_eval)

    for i in range(29):
        p1 = Population(size=4, generation_num=50)
        p1_max_fitness_in_each_eval = p1_max_fitness_in_each_eval + np.array(p1.max_fitness_in_each_eval)

        p2 = Population(size=8, generation_num=25)
        p2_max_fitness_in_each_eval = p2_max_fitness_in_each_eval + np.array(p2.max_fitness_in_each_eval)

        p3 = Population(size=25, generation_num=8)
        p3_max_fitness_in_each_eval = p3_max_fitness_in_each_eval + np.array(p3.max_fitness_in_each_eval)

    x = [i for i in range(len(p1_max_fitness_in_each_eval))]
    plt.plot(x, p1_max_fitness_in_each_eval / 30, label='size=4')
    plt.plot(x, p2_max_fitness_in_each_eval / 30, label='size=8')
    plt.plot(x, p3_max_fitness_in_each_eval / 30, label='size=25')
    plt.legend()
    plt.title('Average performance of 30 runs')
    plt.xlabel('evaluation times')
    plt.ylabel('best so far fitness')
    plt.show()


if __name__ == '__main__':
    main()
