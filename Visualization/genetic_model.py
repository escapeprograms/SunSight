from math import floor
from torch import nn
import torch

from Neat.evaluation_util import *


# Idea 1:
# feed in realized potential, EGP, and COP for every single zip code, and have all unchanging demographic information learned by weights
# problem: massive input, massive output
# Idea 2:
# feed in ONLY realized potential of each demographic. Let the weights learn which zips are in which demographic, which zips are good for COP and EGP

class Predictor(nn.Module):
    def __init__(self, input_len, hidden_size, output_size):
        super(Predictor, self).__init__()
        self.input_len = input_len
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lin1 = nn.Linear(self.input_len, self.hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.lin2 = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax()
    
    def forward(self, X):
        out = self.lin1(X)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.softmax(out)
        return out
    

#initialize models

#mutate and reproduce
def mutate(model, mutation_rate = 0.1, noise_magnitude = 0.1):
    for param in model.parameters():
        # Generate a mask for which weights to mutate
        mutation_mask = (torch.rand(param.size()) < mutation_rate).float()
        
        # Generate Gaussian noise and apply it where the mask is 1
        noise = torch.randn_like(param) * noise_magnitude
        param.data += mutation_mask * noise #only add noise to the unmasked weights

            
def propogate_model(model, df, num_children):
    children = []
    for i in range(num_children):
        child_model = Predictor(2, 10, df.shape[0]) #TODO: load a model w/ same params
        child_model.load_state_dict(model.state_dict())
        #mutate all models except first model
        if i != 0:
            mutate(child_model)
        children.append(child_model)
    return children

def propogate(simulations, df, num_children):
    new_models = []
    for sim in simulations:
        model = sim.model
        children = propogate_model(model, df, int(num_children))
        for child in children:
            new_models.append(child)
    return new_models

#train a single generation
def train_generation(models, df, num_panels, survivor_fraction=0.1):
    #"count_qualified" is the maximum panels allowed in a zip/state
    #"existing_installs_count" is the current panels allowed in a zip/state
    simulations = [SimulationState(df, models[j]) for j in range(len(models))]
    score_sim_pairs = [] #list of (score, simulation)

    for j in range(len(simulations)):
        for i in range(num_panels):
            simulations[j].step()
        print("simulations done:",j)
        score_sim_pairs.append((simulations[j].score(), simulations[j]))

    #perform natural selection and return models
    score_sim_pairs.sort(key=lambda s: s[0], reverse=True) #sort by highest score
    num_survivors = floor(survivor_fraction * len(simulations))
    successful_models = [score_sim_pairs[i][1] for i in range(num_survivors)]

    print("Generation finished! High score:", score_sim_pairs[0][0])
    return successful_models

#train over many generations
def train(df, num_panels=1000, num_models=100, generations=200, survivor_fraction = 0.1):
    models = [Predictor(2, 10, df.shape[0]) for i in range(num_models)]
    for i in range(generations):
        print("Starting generation",i)
        survivors = train_generation(models, df, num_panels, survivor_fraction=survivor_fraction)
        models = propogate(survivors, df, floor(1/survivor_fraction))

    return survivors[0].model #return the best model by the end
#TODO: test all of these with a real test run