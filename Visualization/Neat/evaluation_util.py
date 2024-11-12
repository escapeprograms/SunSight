from collections import Counter
import copy
import numpy as np
import pandas as pd
import torch


def get_demo_split(df, demos, key, type="avg value"):
    true_avg = np.mean(df[key].values)

    new_df = pd.DataFrame()
    low_avgs = []
    high_avgs = []
    
    for demo in demos:
        median = np.median(df[demo].values)
        low_avg = np.mean(df[df[demo] < median][key].values)
        high_avg = np.mean(df[df[demo] >= median][key].values)

        if type == "percent":
            low_avg = ((low_avg/true_avg) - 1) * 100
            high_avg = ((high_avg/true_avg) -1) * 100
        if type == "diff":
            low_avg = true_avg - low_avg
            high_avg = true_avg - high_avg
        if type == 'paper':
            low_avg /= true_avg
            high_avg /= true_avg

        low_avgs.append(low_avg)
        high_avgs.append(high_avg)

    new_df['demographic'] = demos
    new_df['Below median'] = low_avgs
    new_df['Above median'] = high_avgs

    # print("-------")
    print(new_df)
    n = new_df.to_numpy()
    return n[:,1:].flatten()
    

# store simulation state
class SimulationState:
    def __init__(self, init_df):
        self.installs = init_df['existing_installs_count'].to_numpy(copy=True) #df of all zips/states and their installation count

        realized_potential = init_df['realized_potential_percent'].to_numpy() #Note: this is not actually a percent, but a ratio
        self.install_capacity = self.installs / realized_potential #maximum capacity of installations per state

        #count_qualified not defined in state_df
        #alternate state install capacity method: sum up count_qualified for every zip code in a state
        # self.install_capacity = init_df['count_qualified'].to_numpy() #maximum capacity of installations per zip

        # binary mask for whether each zip is above/below each equity factor
        black_median = init_df['black_prop'].median()
        self.black_mask = (init_df['black_prop'] > black_median).astype(int).to_numpy()

        income_median = init_df['Median_income'].median()
        self.income_mask = (init_df['Median_income'] > income_median).astype(int).to_numpy()

    #TODO: gather network inputs for the model
    def network_inputs(self):
        #we want to pass in the realized potential for 50 states (and DC)
        #len = 51
        return self.installs / self.install_capacity
    # can we add more panels here
    def at_capacity(self, zip):
        return self.install_capacity[zip] < self.installs[zip]
    
    # add panel based on the NN output
    def add_panels(self, pred, num_panels=1):
        ranking = [(pred[i], i) for i in range(len(pred))]
        ranking.sort(key=lambda x: x[0], reverse=True) #rank panels in decreasing order
        
        #add num_panels panels to the simulation
        j = 0
        for i in range(num_panels):
            while self.at_capacity(ranking[j][1]): #find the first prefered zip code that can accomodate a new panel
                j += 1
            self.installs[ranking[j][1]] += 1

    #TODO: calculate score with lexicase
    def score(self):
        #mode = np.floor(np.random.rand() * 5)
        mode = 0
        if mode == 0: # judge based on racial equity
            return self.score_racial_equity()
        elif mode == 1: # judge based on income equity
            return self.score_income_equity()
        elif mode == 2: # judge based on geographic equity
            pass
        elif mode == 3: # judge based on carbon offset
            pass
        elif mode == 4: # judge based on energy generation
            pass
    
    def score_racial_equity(self):
        # basically sum the installs in black zip codes
        x = np.dot(self.black_mask, self.installs)
        return x

    def score_income_equity(self):
        x = np.dot(self.income_mask, self.installs)
        return x