from collections import Counter
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
    

# store evaluation state
class SimulationState:
    def __init__(self, init_df, model):
        self.model = model #the neural network being trained

        self.installs = init_df['existing_installs_count'].to_numpy() #df of all zips/states and their installation count
        self.install_capacity = init_df['count_qualified'].to_numpy() #maximum capacity of installations per zip

        # binary mask for whether each zip is above/below each equity factor
        black_median = init_df['black_prop'].median()
        self.black_mask = (init_df['black_prop'] > black_median).astype(int).to_numpy()

        income_median = init_df['Median_income'].median()
        self.income_mask = (init_df['Median_income'] > income_median).astype(int).to_numpy()

    def step(self):
        #pass in racial equity, income equity, TODO: include generated energy, carbon offset, location equity
        realized_potential_race = self.score_racial_equity() / np.dot(self.black_mask, self.install_capacity)
        realized_potential_income = self.score_income_equity() / np.dot(self.income_mask, self.install_capacity)

        pred = self.model(torch.Tensor([realized_potential_race, realized_potential_income])) #predict model based on simulation state

        #post-process and add a panel
        ranking = [(pred[i], i) for i in range(pred.shape[0])]
        ranking.sort(key=lambda x: x[0], reverse=True) #rank panels in decreasing order
        
        j = 0
        while self.at_capacity(ranking[j][1]): #find a zip that can accomodate a new panel
            j += 1
        self.add_panel(ranking[j][1])

    # can we add more panels here
    def at_capacity(self, zip):
        return self.install_capacity[zip] < self.installs[zip]
    
    # add panel
    def add_panel(self, zip):
        self.installs[zip] += 1
    
    #calculate scores
    def score(self):
        #mode = np.floor(np.random.rand() * 5)
        mode = 0
        if mode == 0: # judge based on racial equity
            return self.score_racial_equity()
        elif mode == 1: # judge based on income equity
            pass
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