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
    
#data manager

class DataManager:
    def __init__(self, combined_df, state_df,
                 fields=['Republican_prop', 'Median_income', 'carbon_offset_kg_per_panel', 'energy_generation_per_panel', 'realized_potential_percent', 'black_prop']):
        #Note: remove Republican_prop
        #fields needs to consider energy efficiency, equity, and carbon efficiency
        self.combined_df = combined_df
        self.state_df = state_df
        self.num_zips = len(self.combined_df)
        self.fields = fields
        self.synthesize_df()

    def synthesize_df(self): #create a full df per zip code
        #change the Washington, D.C. key in state_df to be compatible with combined_df
        self.state_df.loc[47, 'State'] = "District of Columbia"

        #Merge political preference into combined_df
        republian_prop = self.state_df[['State', 'Republican_prop']]
        self.combined_df = self.combined_df.merge(republian_prop, left_on='state_name', right_on='State', how='left')

        #only use desired fields
        # self.combined_df = self.combined_df[self.fields]

        #normalize all inputs to [0,1]
        self.combined_df[self.fields] = (self.combined_df[self.fields] - self.combined_df[self.fields].min()) / (self.combined_df[self.fields].max() - self.combined_df[self.fields].min())

        
        
    #get all data associated with a zip code
    def get_zip(self, id): 
        return self.combined_df.loc[id, self.fields].tolist()

    #return feature-engineered vector for a zip code
    def network_inputs(self, id):
        #add bias
        inputs = self.get_zip(id) + [1]
        return inputs
    
    #greedily project based on an arbitrary order
    def greedy_projection(self, zip_order, metric='carbon_offset_metric_tons_per_panel', n=1000, record=False):
        projection = np.zeros(n+1)
        ordered_index = 0
        greedy_index = zip_order[ordered_index]
        existing_count = self.combined_df['existing_installs_count'][greedy_index]
        i = 0 #panel counter

        picked = []
        if record:
            picked = [self.combined_df['region_name'][greedy_index]]

        while (i < n):
            if existing_count >= self.combined_df['count_qualified'][greedy_index]: # this location is full
                ordered_index += 1
                greedy_index = zip_order[ordered_index]

                existing_count = self.combined_df['existing_installs_count'][greedy_index]

            else:
                projection[i+1] = projection[i] + self.combined_df[metric][greedy_index] #add a new panel to this location
                existing_count += 1
                i += 1
                if record:
                    picked.append(self.combined_df['region_name'][greedy_index]) #record every panel location in order
        
        return projection, picked
    
    #TODO: FIX THE METRICS
    def score(self, zip_order, mode = 0, n = 1000, record=False):
        # mode = 3
        #TODO: lexicase on 2 objectives: score carbon offset and score energy generation
        if mode == "geographic_equity": # judge based on geographic equity
            return self.score_geographic_equity(zip_order, n)
        elif mode == "racial_equity": # judge based on racial equity
            return self.score_racial_equity(zip_order, n)
        elif mode == "income_equity": # judge based on income equity
            return self.score_income_equity(zip_order, n)
        elif mode == "carbon_offset": # judge based on carbon offset
            return self.score_carbon_offset(zip_order, n)
        elif mode == "energy_generation": # judge based on energy generation
            return self.score_energy_generation(zip_order, n)
    
    def score_racial_equity(self, zip_order, n=1000):
        #TODO: find a better way to do this
        score, _ = self.greedy_projection(zip_order, 'black_prop', n)
        return score[-1]

    def score_income_equity(self, zip_order, n=1000):
        #TODO: Make this actually score equity
        score, _ = self.greedy_projection(zip_order, 'Median_income', n)
        return score[-1]
    
    def score_geographic_equity(self, zip_order, n=1000):
        #TODO: IMPLEMENT THIS METRIC
        score, _ = [0]
        return score[-1]

    def score_carbon_offset(self, zip_order, n=1000):
        score, _ = self.greedy_projection(zip_order, 'carbon_offset_kg_per_panel', n)
        return score[-1]

    def score_energy_generation(self, zip_order, n=1000):
        score, _ = self.greedy_projection(zip_order, 'energy_generation_per_panel', n)
        return score[-1]
    

# store simulation state - DEPRECATED
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