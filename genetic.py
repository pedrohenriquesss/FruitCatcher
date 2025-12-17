import random

def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]

def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations, elite_rate=0.2, mutation_rate=0.05):
    population = generate_population(individual_size, population_size)
    best_individual = None
    
    # Complete the algorithm
    best_fitness = -1

    for generation in range(generations):
        evaluated_population = [(individual, fitness_function(individual, seed=generation)) for individual in population]
        evaluated_population.sort(key=lambda item: item[1], reverse=True) # população ordenada pelo seu fitness, os melhores são os primeiros índices

        current_best, current_fitness = evaluated_population[0]

        if current_fitness > best_fitness:
            best_individual = (current_best.copy(), current_fitness)
            best_fitness = current_fitness

        print(f"Generation {generation}: Best fitness = {best_fitness}")

        if best_fitness >= target_fitness:
            break

        # cria a próxima geração
        elite_size = int(elite_rate * population_size)
        next_generation = [individual for individual, _ in evaluated_population[:elite_size]]

        # os restantes membros da próxima geração são os filhos
        while len(next_generation) < population_size:
            parent1 = select_parent(evaluated_population)
            parent2 = select_parent(evaluated_population)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            next_generation.append(child)

        population = next_generation

    return best_individual # This is expected to be a pair (individual, fitness)

def select_parent(evaluated_population):
    candidates = random.sample(evaluated_population, k=2)  # seleciona 2 aleatórios
    # cada elemento é (individual, fitness)
    best_candidate = max(candidates, key=lambda item: item[1])  # item[1] = fitness
    return best_candidate[0]  # retorna o individual, não o fitness

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += random.uniform(-0.2, 0.2)  # perturbação suave
            individual[i] = max(min(individual[i], 1), -1)  # mantém em [-1, 1]