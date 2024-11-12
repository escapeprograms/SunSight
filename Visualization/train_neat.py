# cd Visualization

import os
import neat
from Neat.evaluation_util import SimulationState
from data_load_util import load_state_data, make_dataset

#constants
NUM_PANELS = 1000
PANEL_BATCH_SIZE = 100

#evaluate a simulation
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        simulation = SimulationState(state_df)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i in range(0, NUM_PANELS, PANEL_BATCH_SIZE):
            pred = net.activate(simulation.network_inputs())
            simulation.add_panels(pred, num_panels=PANEL_BATCH_SIZE)
        genome.fitness = simulation.score()
        print("***finished simulation", genome_id)


def run(config_file):
    print("loading configuration...")
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    
    print("creating population...") #WARNING: population takes a LONG time to create
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    print("setting reporters...")
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, filename_prefix='Neat/checkpoints/neat-checkpoint-'))

    # Run for up to 300 generations.
    print("training model...")
    winner = p.run(eval_genomes, 3)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'Neat/neat-config')

    
    #load datasets
    combined_df = make_dataset(remove_outliers=True)
    state_df = load_state_data(combined_df, load="Clean_Data/data_by_state.csv")
    print("Running Genetic Algorithm")
    run(config_path)