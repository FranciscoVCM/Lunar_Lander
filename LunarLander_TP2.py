import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
TEST_EPISODES = 1000
STEPS = 500

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100
PROB_CROSSOVER = 0.9 #testamos tambem a 0.5

PROB_MUTATION = 1.0/GENOTYPE_SIZE
STD_DEV = 0.1 # testamos a 0.05

TOURNAMENT_SIZE = 3 # testamos a 5

ELITE_SIZE = 1

def network(shape, observation,ind):
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x

def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        return True
    return False

def objective_function(observation_history):

    final_observation = observation_history[-1]

    x = final_observation[0]
    y = final_observation[1]
    vx = final_observation[2]
    vy = final_observation[3]
    theta = final_observation[4]
    vtheta = final_observation[5]
    contact_left = final_observation[6]
    contact_right = final_observation[7]

    successful_landing = check_successful_landing(final_observation)

    # distância à zona de aterragem, valorizamos o erro horizontal mais porque o alvo está centrado em exatamente x=0 
    distance_score = -2.0 * abs(x) - 0.5 * abs(y)

    # penaliza movimento demasiado rápido, que pode ser inseguro para uma aterragem segura
    velocity_score = -1.0 * abs(vx) - 1.5 * abs(vy)

    # penaliza landers inclinados ou ainda a rodar 
    attitude_score = -1.0 * abs(theta) - 0.3 * abs(vtheta)

    # Partial reward for touching the ground with the legs. This helps evolution
    # distinguish almost-good landings from totally unstable crashes.

    # atribuir reward para tocar no chão com cada perna 
    leg_contact_score = 2.0 * contact_left + 2.0 * contact_right

    # rewards de estabilidade extra 
    pad_bonus = 5.0 if abs(x) <= 0.2 else 0.0
    stable_velocity_bonus = 5.0 if vy > -0.2 and abs(vx) < 0.2 else 0.0
    stable_orientation_bonus = 5.0 if abs(theta) < np.deg2rad(20) else 0.0

    # reward grande para aterragens bem sucedidas
    success_bonus = 100.0 if successful_landing else 0.0

    fitness = (distance_score + velocity_score + attitude_score +
               leg_contact_score + pad_bonus + stable_velocity_bonus +
               stable_orientation_bonus + success_bonus)

    return fitness, successful_landing

def simulate(genotype, render_mode = None, seed=None, env = None):
    
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
        
    observation, info = env.reset(seed=seed)

    observation_history = [observation]
    for _ in range(STEPS):
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        
        observation_history.append(observation)

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective_function(observation_history)

def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes
    
    env = gym.make("LunarLander-v3", render_mode =None, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
    while True:
        ind = evaluationQueue.get()

        if ind is None:
            break
            
        ind['fitness'] = simulate(ind['genotype'], seed = None, env = env)[0]
                
        evaluatedQueue.put(ind)
    env.close()
    
def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def parent_selection(population):
    # seleção por torneio: seleciona TOURNAMENT_SIZE indivíduos aleatoriamnte e retorna o com fitness mais alta
   
    tournament_size = min(TOURNAMENT_SIZE, len(population))
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=lambda ind: ind['fitness'])
    return copy.deepcopy(winner)

def crossover(p1, p2):
    # crossover em que cada gene da criança é copiado por um dos pais com probabilidade de 0.5
    child_genotype = []
    for g1, g2 in zip(p1['genotype'], p2['genotype']):
        if random.random() < 0.5:
            child_genotype.append(g1)
        else:
            child_genotype.append(g2)

    return {'genotype': child_genotype, 'fitness': None}

def mutation(p):
    # mutação gaussiana, cada gene tem probabilidade PROB_MUTATION de ser perturbado por gaussian noise com desvio padrão STD_DEV
    mutant = copy.deepcopy(p)
    mutant['fitness'] = None

    for i in range(len(mutant['genotype'])):
        if random.random() < PROB_MUTATION:
            mutant['genotype'][i] += random.gauss(0, STD_DEV)

    return mutant    
    
def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution():
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    #Create initial population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)

            else:
                ni = parent_selection(population)
                
            ni = mutation(ni)
            offspring.append(ni)
            
        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        
        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        print(f'Best of generation {gen}: {best[1]}')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return bests

def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests

if __name__ == '__main__':

    #Pick a setting from below
    #--to evolve the controller--    
    #evolve = True
    #render_mode = None

    #--to test the evolved controller without visualisation--
    #evolve = False
    #render_mode = None

    #--to test the evolved controller with visualisation--
    evolve = False
    render_mode = 'human'
    
    
    if evolve:
        #evolve individuals
        n_runs = 5
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
        for i in range(n_runs):    
            random.seed(seeds[i])
            bests = evolution()
            with open(f'log{i}.txt', 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

                
    else:
        #test evolved individuals
        #pick the file to test
        filename = 'log0.txt' #pode-se mudar para os logs entre 0 e 4
        bests = load_bests(filename)
        b = bests[-1]
        SHAPE = b[1]
        ind = b[2]
            
        ind = {'genotype': ind, 'fitness': None}
            
            
        ntests = TEST_EPISODES

        fit, success = 0, 0
        for i in range(1,ntests+1):
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
            fit += f
            success += s
        print(fit/ntests, success/ntests)
