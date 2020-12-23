from individual import Individual
import torch
import torch.nn as nn
from torch import optim

from customized_model import CustomizedModel
from individual import Individual
from tqdm import tqdm

class Population:
    def __init__(self, population_size, device, epoch, verbose=False):
        self.size = population_size
        self.epoch = epoch
        self.device = device
        self.verbose = verbose
        self.individuals = []
        pass

    def initialization(self):
        self.individuals = []
        for i in range(self.size):
            individual = Individual(1000, 10)
            individual.initialization()
            self.individuals.append(individual)

    def evaluation(self, model, classes, dataloader):
        for i, individual in tqdm(enumerate(self.individuals), total=len(self.individuals)):
            # print('result: ', end='')
            fitness = individual.evaluation(model, classes, dataloader, self.device)
            # print(fitness)

    def train(self, trainloader, writer):
        for i, individual in enumerate(self.individuals):
            print('%d individual is training...' % (i+1), end=' ')
            individual.train(trainloader, writer, self.epoch, self.verbose)
            print('done.')

    def eval(self, classes, testloader):
        for i, individual in enumerate(self.individuals):
            print('%d individual is evaluating...' % (i+1), end=' ')
            acc = individual.eval(classes, testloader, self.verbose)
            print(acc, '%')

    def copy(self):
        p = Population(self.size, self.epoch, self.device, self.verbose)
        p.individuals = self.individuals
        return p

    def create_like(self):
        p = Population(self.size, self.epoch, self.device, self.verbose)
        return p

    def setBest(self):
        # Use only after selection
        tmp = [individual for individual in self.individuals if individual.fitness[1] >= 99]
        # tmp = sorted(tmp, key=lambda i: i.fitness[1])
        tmp = sorted(tmp, key=lambda i: i.fitness[0], reverse=True)
        self.best = tmp[0].fitness
        return self.best

    def getFitness(self):
        fitness = []
        accuracy = []
        for i in self.individuals:
            fitness.append(i.fitness[0])
            accuracy.append(i.fitness[1])

        return fitness, accuracy
