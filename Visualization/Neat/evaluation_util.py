from collections import Counter
import numpy as np
import pandas as pd


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


        #save percentile thresholds
        self.racial_thresholds = None
        self.income_thresholds = None

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
    def greedy_projection_accumulator(self, zip_order, metric='carbon_offset_metric_tons_per_panel', n=1000, record=False):
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
                projection[i+1] = projection[i] + self.combined_df[metric][greedy_index] #update accumulator
                
                #add a new panel to this location
                existing_count += 1
                i += 1
                if record:
                    picked.append(self.combined_df['region_name'][greedy_index]) #record every panel location in order
        
        return projection, picked
    
    def greedy_projection_bucket(self, zip_order, metric='carbon_offset_metric_tons_per_panel', n=1000, record=False, thresholds=[]):
        # projection = np.zeros(n+1)
        ordered_index = 0
        greedy_index = zip_order[ordered_index]
        existing_count = self.combined_df['existing_installs_count'][greedy_index]
        i = 0 #panel counter
        bucket = 0 #bucket index
        buckets = [0 for i in range(len(thresholds)-1)]

        picked = []
        if record:
            picked = [self.combined_df['region_name'][greedy_index]]

        while (i < n):
            if existing_count >= self.combined_df['count_qualified'][greedy_index]: # this location is full
                ordered_index += 1
                greedy_index = zip_order[ordered_index]

                existing_count = self.combined_df['existing_installs_count'][greedy_index]

                #set the bucket to add to
                value = self.combined_df[metric][greedy_index]

                for j in range(len(thresholds)):
                    if value > thresholds[j]:
                        bucket = j
            else:
                existing_count += 1
                i += 1
                buckets[bucket] += 1
                
                if record:
                    picked.append(self.combined_df['region_name'][greedy_index]) #record every panel location in order
        
        return buckets, picked
    
    #TODO: FIX THE METRICS
    def score(self, zip_order, mode = "energy_generation", n = 1000, record=False):
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
    
    def score_racial_equity(self, zip_order, n=1000, k=2):
        #get bucket thresholds for percentiles
        if not isinstance(self.racial_thresholds, np.ndarray):
            #only calculate these once; note: k CANNOT change once this is set
            q = np.linspace(0, 1, k+1)
            self.racial_thresholds = self.combined_df['black_prop'].quantile(q).to_numpy()
        else:
            k = len(self.racial_thresholds) - 1

        buckets, _ = self.greedy_projection_bucket(zip_order, 'black_prop', n, thresholds=self.racial_thresholds)

        #get "error": squared relative difference from a uniform distribution
        errors = np.array([(bucket - n/k)/(n/k) for bucket in buckets])

        return 1 - (np.sum(errors**2)/k) #take the negative sum of squared error, normalized to [0,1]

    def score_income_equity(self, zip_order, n=1000, k=2):
        #get bucket thresholds for percentiles
        if not isinstance(self.income_thresholds, np.ndarray):
            #only calculate these once; note: k CANNOT change once this is set
            q = np.linspace(0, 1, k+1)
            self.income_thresholds = self.combined_df['Median_income'].quantile(q).to_numpy()
        else:
            k = len(self.income_thresholds) - 1

        buckets, _ = self.greedy_projection_bucket(zip_order, 'Median_income', n, thresholds=self.income_thresholds)

        #get "error": squared relative difference from a uniform distribution
        errors = np.array([(bucket - n/k)/(n/k) for bucket in buckets])

        return 1 - (np.sum(errors**2)/k) #take the negative sum of squared error, normalized to [0,1]
    
    def score_geographic_equity(self, zip_order, n=1000):
        #TODO: IMPLEMENT THIS METRIC
        score, _ = [0]
        return score[-1]

    def score_carbon_offset(self, zip_order, n=1000):
        score, _ = self.greedy_projection_accumulator(zip_order, 'carbon_offset_kg_per_panel', n)
        return score[-1]/n #normalize [0, 1]

    def score_energy_generation(self, zip_order, n=1000):
        score, _ = self.greedy_projection_accumulator(zip_order, 'energy_generation_per_panel', n)
        return score[-1]/n #normalize [0, 1]
    