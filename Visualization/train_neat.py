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

#constants
NUM_PANELS = 100000
NUM_GENERATIONS = 3
EVALUATION_METRICS = ['racial_equity']#['carbon_offset','energy_generation','racial_equity'] #lexicase metrics to evaluate
OVERALL_THRESHOLD = 0.1 #what fraction of TOTAL population reproduces, makes sure this matches 'survival_threshold' in neat-config

step_threshold = OVERALL_THRESHOLD ** (1/len(EVALUATION_METRICS)) #calculate what fraction survives after each metric is applied sequentially
best_scores = []

#run a simulation
def run_genome(genome, config, data_manager):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    zip_outputs = []
    
    for i in range(0, data_manager.num_zips):
        score = net.activate(data_manager.network_inputs(i))
        zip_outputs.append((i, score))
    zip_outputs.sort(key=lambda z: z[1], reverse=True) #sort by highest score
    zip_order = [index for index, score in zip_outputs]
    return zip_order

#score a simulation and record the cumulative score across all metrics
def score_order(info, eval_metric):
    score = data_manager.score(info[1], eval_metric, NUM_PANELS)
    print("metric:", eval_metric,"score:",score)
    info[2] += score
    return score

#scoring function for all genomes
def eval_genomes(genomes, config):
    best_score = float('-inf') #record best score
    
    #get zip orders for all genomes by running each NN once
    genome_info = [] #stores genome, zip_order, and ranking data
    for genome_id, genome in tqdm(genomes):
        genome.fitness = float('-inf') #set all fitness to a minimum initially
        zip_order = run_genome(genome, config, data_manager)
        genome_info.append([genome, zip_order, 0])

    #lexicase: evaluate based on all metrics in random order
    shuffled_metrics = EVALUATION_METRICS.copy()
    random.shuffle(shuffled_metrics)
    for eval_metric in shuffled_metrics:
        #score each genome's zip order based on eval_metric and sort the list by it
        # genome_info.sort(key = lambda info: data_manager.score(info[1], eval_metric, NUM_PANELS), reverse=True)
        genome_info.sort(key = lambda info: score_order(info, eval_metric), reverse=True)
        
        #naturally select the genome list by the step_threshold
        genome_info = genome_info[0:np.ceil(step_threshold * len(genome_info)).astype(int)]

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

def run(config_file, checkpoint=0):
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
    
    winner = p.run(eval_genomes, NUM_GENERATIONS)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    #save output into a pickle
    with open('Neat/models/NEAT_model.pkl', 'wb') as f:
        pickle.dump(winner_net, f)
    return winner_net


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'Neat/neat-config')

    
    #load datasets
    combined_df = make_dataset(remove_outliers=True)
    state_df = load_state_data(combined_df, load="Clean_Data/data_by_state.csv")
    data_manager = DataManager(combined_df, state_df)
    print("Running Genetic Algorithm")
    run(config_path)


    #save the best scores
    with open('Neat/models/fitness_data.pkl', 'wb') as f:
        pickle.dump(best_scores, f)