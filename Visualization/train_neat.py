# cd Visualization

from datetime import datetime
import os
import pickle
import neat
import numpy as np
from Neat.evaluation_util import DataManager
from data_load_util import load_state_data, make_dataset
from tqdm import tqdm
import random
from Neat.saving_util import *

#constants
NUM_PANELS = 100000
NUM_GENERATIONS = 3
EVALUATION_METRICS = ['carbon_offset','energy_generation','racial_equity','income_equity'] #lexicase metrics to evaluate
METRIC_WEIGHTS = [1,3,1,1] #how much each metric should weight
OVERALL_THRESHOLD = 0.3 #what fraction of TOTAL population reproduces, makes sure this matches 'survival_threshold' in neat-config

step_threshold = OVERALL_THRESHOLD ** (1/sum(METRIC_WEIGHTS)) #calculate what fraction survives after each metric is applied sequentially
best_scores = []

#run a simulation
def run_network(net, data_manager, train=True, cross_val = False):
    zip_outputs = []
    train_ind, test_ind = data_manager.get_fold_indices()

    #cross val vs train
    if cross_val == False:
        indices = train_ind
    else:
        indices = test_ind

    #test set
    if train == False:
        indices = range(data_manager.num_zips)

    for i in indices:
        score = net.activate(data_manager.network_inputs(i, train=True))
        zip_outputs.append((i, score))
    zip_outputs.sort(key=lambda z: z[1], reverse=True) #sort by highest score
    zip_order = [index for index, score in zip_outputs]
    return zip_order

#score a simulation and record the cumulative score across all metrics
def score_order(info, metric_ind):
    #info[1] is zip_order, info[2] is score
    score = data_manager.score(info[1], EVALUATION_METRICS[metric_ind], NUM_PANELS, train=True)
    # print("metric:", eval_metric,"score:",score)
    info[2] += score * METRIC_WEIGHTS[metric_ind]
    return score

#GENOME SELECTION
#lexicase
def eval_genomes_lexicase(genomes, config):
    best_score = float('-inf') #record best score
    
    #get zip orders for all genomes by running each NN once
    genome_info = [] #stores genome, zip_order, and cumulative score
    for genome_id, genome in tqdm(genomes):
        genome.fitness = float('-inf') #set all fitness to a minimum initially
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        zip_order = run_network(net, data_manager)
        genome_info.append([genome, zip_order, 0]) #genome pointer, zip_order, and cumulative score

    #lexicase: evaluate based on all metrics in random order
    indices = np.random.permutation(len(EVALUATION_METRICS))
    for i in indices:
        eval_metric = EVALUATION_METRICS[i]
        metric_weight = METRIC_WEIGHTS[i]
        #score each genome's zip order based on eval_metric and sort the list by it
        # genome_info.sort(key = lambda info: data_manager.score(info[1], eval_metric, NUM_PANELS), reverse=True)
        genome_info.sort(key = lambda info: score_order(info, i), reverse=True)
        
        #naturally select the genome list by the step_threshold
        cutoff = np.ceil((step_threshold**metric_weight * len(genome_info))).astype(int)
        genome_info = genome_info[0:cutoff]

        #update ranking data for tiebreakers, a number closer to 0 is better (deprecated)
        # for i in range(len(genome_info)):
        #     genome_info[i][2] -= i
        
    #set tie-breaker fitness for final survivors
    for genome, zip_order, score in genome_info:
        genome.fitness = score #note: this score can go up to NUM_PANELS * len(EVALUATION_METRICS)

        #find the best performing score in this generation
        if genome.fitness > best_score:
            best_score = genome.fitness

    #mark the best score of this generation
    best_scores.append(best_score)


#fitness proportion
def eval_genomes_fitness_prop(genomes, config):
    best_score = float('-inf') #record best score
    
    #get zip orders for all genomes by running each NN once
    genome_info = [] #stores genome, zip_order, and cumulative score
    for genome_id, genome in tqdm(genomes):
        genome.fitness = 0 #set all fitness to a 0 initially
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        zip_order = run_network(net, data_manager)
        genome_info.append([genome, zip_order, 0]) #genome pointer, zip_order, and cumulative score

    #fitness prop: evaluate based on sum of all metrics (weighted)
    indices = np.arange(len(EVALUATION_METRICS))
    for i in indices:
        for j in range(len(genome_info)):
            score_order(genome_info[j], i)
        
    #set the fitnesses for all genomes
    for genome, zip_order, score in genome_info:
        genome.fitness = score #note: this score can go up to NUM_PANELS * len(EVALUATION_METRICS)

        #find the best performing score in this generation
        if genome.fitness > best_score:
            best_score = genome.fitness

    #mark the best score of this generation
    best_scores.append(best_score)

#tournament selection TODO
def eval_genomes_tournament(genomes, config):
    best_score = float('-inf') #record best score
    
    #get zip orders for all genomes by running each NN once
    genome_info = [] #stores genome, zip_order, and cumulative score
    for genome_id, genome in tqdm(genomes):
        genome.fitness = 0 #set all fitness to a 0 initially
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        zip_order = run_network(net, data_manager)
        genome_info.append([genome, zip_order, 0]) #genome pointer, zip_order, and cumulative score

    #fitness prop: evaluate based on sum of all metrics (weighted)
    indices = np.arange(len(EVALUATION_METRICS))
    for i in indices:
        for j in range(len(genome_info)):
            score_order(genome_info[j], i)
        
    #set the fitnesses for all genomes
    for genome, zip_order, score in genome_info:
        genome.fitness = score #note: this score can go up to NUM_PANELS * len(EVALUATION_METRICS)

        #find the best performing score in this generation
        if genome.fitness > best_score:
            best_score = genome.fitness

    #mark the best score of this generation
    best_scores.append(best_score)

#random selection
def eval_genomes_random(genomes, config):
    for genome_id, genome in tqdm(genomes):
        genome.fitness = 1 #set all fitness to a 1 and let the model do its thing
        

#do a single training run
def run(config_file, selection_method, checkpoint=0):
    print("loading configuration...")
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    
    print("creating population...") #WARNING: population takes a LONG time to create
    if checkpoint == 0:
        p = neat.Population(config)
    else:
        p = neat.Checkpointer.restore_checkpoint(f'Neat/checkpoints/neat-checkpoint-{checkpoint}')

    # Add a stdout reporter to show progress in the terminal.
    print("setting reporters...")
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(time_interval_seconds=1200, filename_prefix='Neat/checkpoints/neat-checkpoint-'))

    # Run for up to 300 generations.
    print("training model...")
    
    winner = p.run(selection_method, NUM_GENERATIONS)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    return winner_net

#do a full K-fold runs
def K_fold_run(config_path, data_manager, selection_method, k=5):
    data_manager.generate_folds(k)

    scores = []
    networks = []
    for i in range(k):
        print(f"running fold {i}")
        data_manager.set_fold(i)
        winner_net = run(config_path, selection_method)
        networks.append(winner_net)

        #evaluate network on test set
        cv_order = run_network(winner_net, data_manager, cross_val=True)

        IE = data_manager.score(cv_order, 'income_equity', NUM_PANELS, train=True)
        RE = data_manager.score(cv_order, 'racial_equity', NUM_PANELS, train=True)
        CO = data_manager.score(cv_order, 'carbon_offset', NUM_PANELS, train=True)
        EG = data_manager.score(cv_order, 'energy_generation', NUM_PANELS, train=True)

        scores.append([IE, RE, CO, EG])
    #average scores
    scores_np = np.array(scores)
    return networks[0], np.mean(scores_np, axis=0)
    

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'Neat/neat-config')

    #load datasets
    combined_df = make_dataset(remove_outliers=True)
    state_df = load_state_data(combined_df, load="Clean_Data/data_by_state.csv")
    data_manager = DataManager(combined_df, state_df)

    #run lexicase
    # lexi_network, lexi_results = K_fold_run(config_path, data_manager, eval_genomes_lexicase, k=5)
    # save_model(lexi_network, lexi_results, model_name="NEAT_model_lexicase.pkl", results_name="lexicase_results.pkl")
    
    #run fitness prop
    fp_network, fp_results = K_fold_run(config_path, data_manager, eval_genomes_fitness_prop, k=5)
    save_model(fp_network, fp_results, model_name="NEAT_model_fitness_prop.pkl", results_name="fitness_prop_results.pkl")

    #tournament selection
    # tourney_network, tourney_results = K_fold_run(config_path, data_manager, eval_genomes_tournament, k=5)
    # save_model(tourney_network, model_name="NEAT_model_tournament.pkl")


    #run random selection
    rand_network, rand_results = K_fold_run(config_path, data_manager, eval_genomes_random, k=5)
    save_model(rand_network, model_name="NEAT_model_random.pkl")

    #print results
    result_metrics = ['income_equity', 'racial_equity', 'carbon_offset', 'energy_generation']
    # for i, res in enumerate(lexi_results):
    #     print(f"Lexicase {result_metrics[i]}", res)
    
    for i, res in enumerate(fp_results):
        print(f"Fitness prop {result_metrics[i]}", res)

    for i, res in enumerate(rand_results):
        print(f"Random {result_metrics[i]}", res)
    
    # data_manager.train_test_split(test_size = 0.0) #no test set for now
    # data_manager.generate_folds(5)
    # data_manager.set_fold(0)

    # print("Running Genetic Algorithm")
    # winner_net = run(config_path)

    # #evaluate winner against the test set
    # cv_order = run_network(winner_net, data_manager, cross_val=True)

    # print("Carbon Offset score: ",data_manager.score(cv_order, 'carbon_offset', NUM_PANELS, train=True))

    # print("Energy Generation score: ",data_manager.score(cv_order, 'energy_generation', NUM_PANELS, train=True))
    
    # #save output into a pickle
    # with open('Neat/models/NEAT_model.pkl', 'wb') as f:
    #     pickle.dump(winner_net, f)

    # #save the best scores
    # with open('Neat/models/fitness_data.pkl', 'wb') as f:
    #     pickle.dump(best_scores, f)