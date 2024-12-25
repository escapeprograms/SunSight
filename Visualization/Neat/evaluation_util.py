from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import warnings


#data manager
class DataManager:
    def __init__(self, combined_df, state_df,
                 fields=['Median_income', 'carbon_offset_kg_per_panel', 'energy_generation_per_panel', 'realized_potential_percent', 'black_prop']):
        #Note: remove Republican_prop
        #fields needs to consider energy efficiency, equity, and carbon efficiency
        self.combined_df = combined_df
        self.state_df = state_df
        self.num_zips = len(self.combined_df)
        self.num_train_zips = self.num_zips # number of training points
        self.fields = fields
        self.synthesize_df()


        #save percentile thresholds
        self.racial_thresholds = None
        self.income_thresholds = None

        #for polynomial basis expansion for network inputs (unused)
        self.poly = PolynomialFeatures(degree=1)

    def synthesize_df(self): #create a full df per zip code
        #change the Washington, D.C. key in state_df to be compatible with combined_df
        self.state_df.loc[47, 'State'] = "District of Columbia"

        #Merge political preference into combined_df
        republian_prop = self.state_df[['State', 'Republican_prop']]
        self.combined_df = self.combined_df.merge(republian_prop, left_on='state_name', right_on='State', how='left')

        #only use desired fields
        # self.combined_df = self.combined_df[self.fields]

        #normalize all inputs to [0,1] in new df
        self.normalized_df = (self.combined_df[self.fields] - self.combined_df[self.fields].min()) / (self.combined_df[self.fields].max() - self.combined_df[self.fields].min())
        self.normalized_df['State'] = self.combined_df['State']

        #add existing installs count as not normalized
        self.normalized_df['existing_installs_count'] = self.combined_df['existing_installs_count'] 

        #by default set the training set to all data
        self.train_df = self.normalized_df
        self.test_df = pd.DataFrame()

        #k_fold validation stuff
        self.k_folds = []
        self.fold_num = 0
    
    #train-test split; call this once
    def train_test_split(self, test_size=0.2, random_state=69):
        self.train_df, self.test_df = train_test_split(self.normalized_df, test_size=test_size, random_state=random_state)

    #generate k folds of data from the train data; call this once
    def generate_folds(self, k=5, random_state=69):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        self.k_folds = []

        for train_idx, test_idx in skf.split(self.train_df, self.train_df['State']):
            self.k_folds.append((train_idx, test_idx))

    #get the indices of a specific fold
    def get_fold_indices(self, num=None):
        #handle case where k folds haven't been set
        if len(self.k_folds) == 0:
            # warnings.warn("No folds has been set. Run generate_folds() before calling get_fold_indices().")
            return range(self.num_zips), []
        
        if num == None:
            num = self.fold_num #default to the currently selected fold
        
        return self.k_folds[num] #returns ([train indices], [test indices])
    
    #select a new fold
    def set_fold(self, num):
        self.fold_num = num


    #get all data associated with a zip code
    def get_zip(self, ind, train=True):
        if train == False:
            return self.normalized_df.loc[ind, self.fields].tolist()
        else:
            return self.train_df.iloc[ind][self.fields].tolist()

    #return feature-engineered vector for a zip code
    def network_inputs(self, id, train=True):
        #perform polynomial basis expansion
        inputs = self.poly.fit_transform([self.get_zip(id, train)])[0]
        return inputs.tolist()
    
    #greedily project based on an arbitrary order
    def greedy_projection_accumulator(self, zip_order, metric='carbon_offset_metric_tons_per_panel', n=1000, record=False, train=True):
        #select the dataframe based on test/train
        if train == False:
            df = self.normalized_df
        else:
            df = self.train_df
        projection = np.zeros(n+1)
        ordered_index = 0
        greedy_index = zip_order[ordered_index]
        existing_count = self.combined_df['existing_installs_count'][greedy_index]
        i = 0 #panel counter
        
        value = df.iloc[greedy_index][metric]

        picked = []
        if record:
            picked = [self.combined_df['region_name'][greedy_index]]

        while (i < n):
            if existing_count >= self.combined_df['count_qualified'][greedy_index]: # this location is full
                ordered_index += 1
                greedy_index = zip_order[ordered_index]

                existing_count = self.combined_df['existing_installs_count'][greedy_index]

                value = df.iloc[greedy_index][metric]
            else:
                projection[i+1] = projection[i] + value #update accumulator (with normalized value)
                
                #add a new panel to this location
                existing_count += 1
                i += 1
                if record:
                    picked.append(self.combined_df['region_name'][greedy_index]) #record every panel location in order
        return projection, picked
    
    def greedy_projection_bucket(self, zip_order, metric='carbon_offset_metric_tons_per_panel', n=1000, record=False, thresholds=[], train=True):
        #TODO: account for all existing panels
        
        #select the dataframe based on test/train
        if train == False:
            df = self.normalized_df
        else:
            df = self.train_df

        ordered_index = 0
        greedy_index = zip_order[ordered_index]
        existing_count = self.combined_df['existing_installs_count'][greedy_index]
        i = 0 #panel counter
        bucket = 0 #bucket index
        # buckets = [0 for i in range(len(thresholds)-1)]
        #add pre-existing buckets first
        buckets = self.existing_bucket(metric, thresholds, train=train)

        #set the bucket initially
        value = df.iloc[greedy_index][metric]
        for j in range(len(thresholds)):
            if value > thresholds[j]:
                bucket = j


        picked = []
        if record:
            picked = [self.combined_df['region_name'][greedy_index]]

        while (i < n):
            if existing_count >= self.combined_df['count_qualified'][greedy_index]: # this location is full
                ordered_index += 1
                greedy_index = zip_order[ordered_index]

                existing_count = self.combined_df['existing_installs_count'][greedy_index]

                #set the bucket to add to (use normalized value)
                value = df.iloc[greedy_index][metric]

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
    
    #count existing panels into buckets
    def existing_bucket(self, metric='black_prop', thresholds=[], train=True):
        #select the dataframe based on test/train
        if train == False:
            df = self.normalized_df
        else:
            df = self.train_df

        #add to buckets
        df[f'{metric}_bucket'] = pd.cut(
        df[metric],
        bins=thresholds
        )

        # Group by IncomeBucket and sum SolarPanels
        buckets = df.groupby(f'{metric}_bucket')['existing_installs_count'].sum()
        return buckets.tolist()
    
    #score the zip order based on various metrics
    def score(self, zip_order, mode = "energy_generation", n = 1000, record=False, train=True):
        # if mode == "geographic_equity": # judge based on geographic equity
        #     return self.score_geographic_equity(zip_order, n)
        if mode == "racial_equity": # judge based on racial equity
            return self.score_racial_equity(zip_order, n, train=train)
        elif mode == "income_equity": # judge based on income equity
            return self.score_income_equity(zip_order, n, train=train)
        elif mode == "carbon_offset": # judge based on carbon offset
            return self.score_carbon_offset(zip_order, n, train=train)
        elif mode == "energy_generation": # judge based on energy generation
            return self.score_energy_generation(zip_order, n, train=train)
    
    def score_racial_equity(self, zip_order, n=1000, k=2, train=True):
        #NOTE: only works with k=2
        #get bucket thresholds for percentiles
        if not isinstance(self.racial_thresholds, np.ndarray):
            #only calculate these once; note: k CANNOT change once this is set
            q = np.linspace(0, 1, k+1)
            self.racial_thresholds = self.normalized_df['black_prop'].quantile(q).to_numpy() #use population quantiles
        else:
            k = len(self.racial_thresholds) - 1

        buckets, _ = self.greedy_projection_bucket(zip_order, 'black_prop', n, thresholds=self.racial_thresholds, train=train)

        total_panels = sum(buckets)
        #get "error": squared relative difference from a uniform distribution (each bucket has total_panels/k)
        # errors = np.array([(bucket - total_panels/k)/(total_panels/k) for bucket in buckets])
        # return 1 - (np.sum(errors**2)/k) #take the negative sum of squared error, normalized to [0,1]

        diff = abs(buckets[1]-buckets[0])
        return 1 - (diff/total_panels)

    def score_income_equity(self, zip_order, n=1000, k=2, train=True):
        #NOTE: only works with k=2
        #get bucket thresholds for percentiles
        if not isinstance(self.income_thresholds, np.ndarray):
            #only calculate these once; note: k CANNOT change once this is set
            q = np.linspace(0, 1, k+1)
            self.income_thresholds = self.normalized_df['Median_income'].quantile(q).to_numpy() #use population quantiles
        else:
            k = len(self.income_thresholds) - 1

        buckets, _ = self.greedy_projection_bucket(zip_order, 'Median_income', n, thresholds=self.income_thresholds, train=train)

        total_panels = sum(buckets)

        #get "error": squared relative difference from a uniform distribution (each bucket has total_panels/k)
        # errors = np.array([(bucket - total_panels/k)/(total_panels/k) for bucket in buckets])

        # return 1 - (np.sum(errors**2)/k) #take the negative sum of squared error, normalized to [0,1]

        diff = abs(buckets[1]-buckets[0])
        return 1 - (diff/total_panels)
    

    def score_carbon_offset(self, zip_order, n=1000, train=True):
        score, _ = self.greedy_projection_accumulator(zip_order, 'carbon_offset_kg_per_panel', n)
        return score[-1]/n #normalize [0, 1]

    def score_energy_generation(self, zip_order, n=1000, train=True):
        score, _ = self.greedy_projection_accumulator(zip_order, 'energy_generation_per_panel', n)
        return score[-1]/n #normalize [0, 1]
    