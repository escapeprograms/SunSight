# cd Visualization

from datetime import datetime
import os
import pickle
import neat
import numpy as np
from Neat.evaluation_util import DataManager
from data_load_util import load_state_data, make_dataset
from tqdm import tqdm

#constants
NUM_PANELS = 100000
NUM_GENERATIONS = 150

best_scores = []
#evaluate a simulation
def eval_genomes(genomes, config):
    # eval_mode = np.floor(np.random.rand() * 4) + 1 #randomly select an evaluation mode from [1,4] (0 is geo which is not implemented)
    eval_mode = 2

    for genome_id, genome in tqdm(genomes):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        zip_outputs = []
        best_score = 0
        for i in range(0, data_manager.num_zips):
            score = net.activate(data_manager.network_inputs(i))
            zip_outputs.append((i, score))
        zip_outputs.sort(key=lambda z: z[1], reverse=True) #sort by highest score
        zip_order = [index for index, score in zip_outputs]

        #add panels greedily and score the model's ranking
        genome.fitness = data_manager.score(zip_order, eval_mode, NUM_PANELS)
        
        #find the best performing score
        if genome.fitness > best_score:
            best_score = genome.fitness

    #mark the score of the best generation
    best_scores.append((eval_mode, best_score))

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
    p.add_reporter(neat.Checkpointer(5, filename_prefix='Neat/checkpoints/neat-checkpoint-'))

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