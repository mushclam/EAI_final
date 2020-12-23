import datetime
from individual import Individual
from genetic_algorithm import GeneticAlgorithm
import os
import time
import argparse

from customized_model import CustomizedModel
from population import Population
import utils

import torch
from torch import optim
from torch import random
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

def main():
    # argument parsing
    parser = argparse.ArgumentParser(description='Evolutionary AI Final Project')
    parser.add_argument('-c', '--config', dest='config_path', type=str, help='Set config file')
    parser.add_argument('-m', '--model', dest='model_path', type=str, help='Set prior model.')
    parser.add_argument('-n', '--test-name', dest='test_name', type=str, help='Set test name')
    parser.add_argument('-v', '--verbose', dest='verbose', type=bool, help='Verbose')
    parser.add_argument('-ps', '--population-size', dest='population_size', type=int, default=50)
    parser.add_argument('-g', '--generation', dest='generation', type=int, help='Generation', default=200)
    parser.add_argument('-l', '--learning-rate', dest='learning_rate', type=float, help='Learning rate', default=1E-4)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, help='Batch size', default=4096)
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, help='Epoch', default=4)
    parser.add_argument('-cp', '--crossover-probability', dest='crossover_prob', type=float, help="Crossover Probability", default=1.0)
    parser.add_argument('-mp', '--mutation-probability', dest='mutation_prob', type=float, help='Mutation Probability', default=0.01)
    args = parser.parse_args()
    print(args)

    # Parameter setup
    population_size = args.population_size
    generation = args.generation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epoch = args.epoch
    crossover_prob = args.crossover_prob
    mutation_prob = args.mutation_prob

    # tensorboard writer
    # writer = SummaryWriter('./runs/eai_exp_')

    # Transforms setup
    transform = transforms.ToTensor()

    # Load train and test dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=8, pin_memory=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=8, pin_memory=True)

    # Set up classes
    classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')

    # image show test
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # utils.imsave(torchvision.utils.make_grid(images))

    # device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = utils.get_model('model1_0.pt')
    model.to(device)

    # max_model = Individual(1000, 10)
    # max_model.initialization(torch.ones([10, 1000]))
    # max_model.evaluation(model, classes, trainloader, device)

    # min_model = Individual(1000, 10)
    # min_model.initialization(torch.zeros([10, 1000]))
    # min_model.evaluation(model, classes, trainloader, device)

    population = Population(population_size, device, epoch)
    population.initialization()
    print('First Evaluation')
    population.evaluation(model, classes, trainloader)

    init_fitness, init_accuracy = population.getFitness()
    plt.scatter(init_fitness, init_accuracy, s=10)
    plt.xlabel('Fitness')
    plt.ylabel('Accuracy')
    plt.savefig('result_0/generation_init.png')

    best_fitness, best_accuracy = population.setBest()
    best = [best_fitness]

    # start_time = time.process_time()
    ga = GeneticAlgorithm(crossover_prob, mutation_prob)

    print('[Genetic Alogrithm Routine Start]')
    for i in range(generation):
        child_population = ga.crossover(population)
        child_population = ga.mutation(child_population)
        child_population.evaluation(model, classes, trainloader)

        population = ga.combination(population, child_population)
        front = ga.fastNonDominatedSort(population)

        population = Population(population_size, device, epoch)

        j = 0
        while len(population.individuals) + len(front[j]) <= population_size:
            # ga.crowdingDistanceAssignment(front[j], min_model, max_model)
            ga.crowdingDistanceAssignment(front[j])
            for individual in front[j]:
                population.individuals.append(individual)
            j += 1
        # ga.crowdingDistanceAssignment(front[j], min_model, max_model)
        ga.crowdingDistanceAssignment(front[j])

        front[j] = sorted(front[j], key=lambda f: f.rank)
        front[j] = sorted(front[j], key=lambda f: f.distance, reverse=True)
        size = len(population.individuals)
        for k in range(population_size - size):
            population.individuals.append(front[j][k])

        pop_fitness, pop_accuracy = population.setBest()
        print('Complete generation [%d]: best: %d, %f' % (i, pop_fitness, pop_accuracy))
        best.append(pop_fitness)

        if pop_fitness > best_fitness:
            best_fitness = pop_fitness
            best_accuracy = pop_accuracy

        # if i % 1 == 1:
        fitness, accuracy = population.getFitness()
        plt.scatter(fitness, accuracy, s=10)
        plt.savefig('result_0/generation_' + str(i) + '.png')

    print('[Routine End]')
    print('[Best Fitness: %d, Best Accuracy: %f]' % (best_fitness, best_accuracy))

    plt.clf()
    end_fitness, end_accuracy = population.getFitness()
    plt.scatter(end_fitness, end_accuracy, s=10)
    plt.xlabel('Fitness')
    plt.ylabel('Accuracy')
    plt.savefig('result_0/generation_final.png')

    plt.scatter(init_fitness, init_accuracy, s=10)
    plt.savefig('result_0/generation_comparison.png')

    plt.clf()
    plt.plot(range(generation + 1), best)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.savefig('result_0/fitness.png')
    
    # total_time = time.process_time() - start_time
    # print('Process time:' + str(total_time) + 'sec')

if __name__ == '__main__':
    main()
