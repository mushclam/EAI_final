from operator import itemgetter
import torch
from population import Population
import random
from individual import Individual
import math

class GeneticAlgorithm():
    def __init__(self, crossover_prob, mutation_prob):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def crossover(self, population:Population):
        random.shuffle(population.individuals)
        tmp_pop = population.create_like()
        tmp_ind = []
        individuals = population.individuals
        for i in range(int(len(individuals)/2)):
            if random.random() <= self.crossover_prob:
                p1_gene = individuals[i].mask.reshape([-1])
                p2_gene = individuals[i + int(len(individuals)/2)].mask.reshape([-1])

                max_length = len(p1_gene)
                point = random.randint(1, max_length)

                tmp = p1_gene[point:]
                p1_gene = torch.cat([p1_gene[:point], tmp]).reshape([10, 1000])
                p2_gene = torch.cat([p2_gene[:point],tmp]).reshape([10, 1000])

                p1 = Individual(1000, 10)
                p1.initialization(p1_gene)
                p2 = Individual(1000, 10)
                p2.initialization(p2_gene)

                tmp_ind.append(p1)
                tmp_ind.append(p2)

        tmp_pop.individuals = tmp_ind

        return tmp_pop

    def mutation(self, population:Population):
        tmp_pop = population.create_like()
        individuals = population.individuals
        for individual in individuals:
            individual.mask = individual.mask.reshape([-1])
            for i, chromosome in enumerate(individual.mask):
                if random.random() <= self.mutation_prob:
                    individual.mask[i] = 1 - chromosome
            individual.mask = individual.mask.reshape([10, 1000])

        tmp_pop.individuals = individuals
        return tmp_pop

    def combination(self, parent:Population, child:Population):
        tmp_pop = Population(parent.size + child.size, parent.device, parent.epoch, parent.verbose)
        tmp_pop.individuals = parent.individuals + child.individuals
        
        return tmp_pop

    def fastNonDominatedSort(self, population:Population):
        front = [[]]
        for i in range(population.size):
            p = population.individuals[i]
            p.dominatedSet = set([])
            p.dominatedCount = 0
            for j in range(population.size):
                if i == j: continue
                q = population.individuals[j]
                if p.dominate(q):
                    p.dominatedSet.add(q)
                elif q.dominate(p):
                    p.dominatedCount += 1

            if p.dominatedCount == 0:
                p.rank = 0
                front[0].append(p)

        i = 0
        while len(front[i]) != 0:
            tmp = []
            for individual in front[i]:
                for d in individual.dominatedSet:
                    d.dominatedCount -= 1
                    if d.dominatedCount == 0:
                        d.rank = i + 1
                        tmp.append(d)
            i += 1
            front.append(tmp)

        return front

    def crowdingDistanceAssignment(self, front, max_f=None, min_f=None):
        # if distance of accuracy make some bias, we need to adjust the min value of accuracy
        if max_f is None and min_f is None:
            max_f = [10000, 100.0]
            min_f = [0, 96.0]
        front_length = len(front)
        fitness_size = len(front[0].fitness)
        for f in front:
            f.distance = 0

        for i in range(fitness_size):
            front = sorted(front, key=lambda f: f.fitness[i])
            front[0].distance = front[front_length-1].distance = math.inf # change fitness_size to length
            for j in range(1, front_length - 1): # to length
                front[j].distance = front[j].distance + (front[j+1].fitness[i] - front[j-1].fitness[i])/abs(max_f[i] - min_f[i])