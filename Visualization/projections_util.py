from collections import Counter
import pickle
from Neat.evaluation_util import DataManager
from data_load_util import *


# Creates a projection of carbon offset if the current ratio of panel locations remain the same 
# allowing partial placement of panels in zips and not accounting in the filling of zip codes.
def create_continued_projection(combined_df, n=1000, metric='carbon_offset_metric_tons'):
    total_panels = np.sum(combined_df['existing_installs_count'])
    # print("total, current existing panels:", total_panels)
    panel_percentage = combined_df['existing_installs_count'] / total_panels
    ratiod_carbon_offset_per_panel = np.sum(panel_percentage * combined_df[metric])
    # x = np.arange(n+1) * ratiod_carbon_offset_per_panel
    # print (x.shape)
    return np.arange(n+1) * ratiod_carbon_offset_per_panel

# Greedily adds 1-> n solar panels to zips which maximize the sort_by metric until no more can be added
# Returns the Carbon offset for each amount of panels added
def create_greedy_projection(combined_df, n=1000, sort_by='carbon_offset_metric_tons_per_panel', ascending=False, metric='carbon_offset_metric_tons_per_panel', record=True):
    sorted_combined_df = combined_df.sort_values(sort_by, ascending=ascending, inplace=False, ignore_index=True)
    projection = np.zeros(n+1)
    greedy_best_not_filled_index = 0
    existing_count = sorted_combined_df['existing_installs_count'][greedy_best_not_filled_index]
    i = 0

    if record:
        picked = [sorted_combined_df['region_name'][greedy_best_not_filled_index]]

    while (i < n):
        if existing_count >= sorted_combined_df['count_qualified'][greedy_best_not_filled_index]: # this location is full
            greedy_best_not_filled_index += 1
            existing_count = sorted_combined_df['existing_installs_count'][greedy_best_not_filled_index]

        else:
            projection[i+1] = projection[i] + sorted_combined_df[metric][greedy_best_not_filled_index] #add a new panel to this location
            existing_count += 1
            i += 1
            if record:
                picked.append(sorted_combined_df['region_name'][greedy_best_not_filled_index]) #record every panel location in order
    
    return projection, picked

def create_continued_equity_projection(combined_df, n=1000, metric='Median_income'):
    #NOTE: only works with thresholds=2; will need to copy DataManager.score_racial_equity to handle k thresholds
    k = 2
    q = np.linspace(0, 1, k+1)
    thresholds = combined_df[metric].quantile(q).to_numpy() 

    total_panels = np.sum(combined_df['existing_installs_count'])

    #look at pre-existing buckets only
    combined_df[f'{metric}_bucket'] = pd.cut(
    combined_df[metric],
    bins=thresholds
    )

    # Group by metric and sum SolarPanels
    buckets = combined_df.groupby(f'{metric}_bucket')['existing_installs_count'].sum().tolist()

    disparity_ratio = (buckets[1]-buckets[0])/total_panels #the equity doesn't change over time
    # print(f"continued {metric} equity ratio:", equity_ratio)
    x = np.arange(total_panels, total_panels + n+1) * disparity_ratio
    # print("final diff:", x[-1])
    return x


def create_equity_projection_from_picked(combined_df, picked = pd.DataFrame(), n=1000, metric='Median_income'):
    #picked is a list of zip codes, we need a list of indices
    #NOTE: only works with k=2; will need to copy DataManager.score_racial_equity to handle k thresholds
    k = 2
    q = np.linspace(0, 1, k+1)
    thresholds = combined_df[metric].quantile(q).to_numpy() 

    ordered_index = 0
    greedy_index = combined_df[combined_df['region_name'] == picked[ordered_index]].index[0] #gets the index of the zip code
    existing_count = combined_df['existing_installs_count'][greedy_index]

    i = 0 #panel counter
    bucket = 0 #bucket index
    # buckets = [0 for i in range(len(thresholds)-1)]
    
    #add pre-existing buckets first
    combined_df[f'{metric}_bucket'] = pd.cut(
    combined_df[metric],
    bins=thresholds
    )

    # Group by metric and sum SolarPanels
    buckets = combined_df.groupby(f'{metric}_bucket')['existing_installs_count'].sum().tolist()

    #set the bucket initially
    value = combined_df.iloc[greedy_index][metric]
    for j in range(len(thresholds)):
        if value > thresholds[j]:
            bucket = j

    equity_projection = np.zeros(n+1)
    equity_projection[0] = (buckets[1]-buckets[0])

    
    # place panels
    for i, zip_code in enumerate(picked):
        #set the bucket to add to (use normalized value)
        if i == 0 or picked[i] != picked[i-1]:
            greedy_index = combined_df[combined_df['region_name'] == zip_code].index[0] #gets the index of the zip code
            value = combined_df.iloc[greedy_index][metric]
            for j in range(len(thresholds)):
                if value > thresholds[j]:
                    bucket = j

        buckets[bucket] += 1
        equity_projection[i] = (buckets[1]-buckets[0]) #find diff between panels in below median zips and panels in above median zips
    return equity_projection

    #broken method
    # while (i < n):
    #     if existing_count >= combined_df['count_qualified'][greedy_index]: # this location is full
    #         ordered_index += 1
    #         greedy_index = combined_df[combined_df['region_name'] == picked[ordered_index]].index[0] #gets the index of the zip code

    #         existing_count = combined_df['existing_installs_count'][greedy_index]

    #         #set the bucket to add to (use normalized value)
    #         value = combined_df.iloc[greedy_index][metric]

    #         for j in range(len(thresholds)):
    #             if value > thresholds[j]:
    #                 bucket = j
    #     else:
    #         buckets[bucket] += 1
    #         equity_projection[i+1] = (buckets[1]-buckets[0]) #find diff between panels in below median zips and panels in above median zips
    #         existing_count += 1
    #         i += 1
    # return equity_projection

# Creates a projection which decides each placement alternating between different policies
def create_round_robin_projection(projection_list, picked_list):
    n = len(projection_list[0])
    number_of_projections = len(projection_list)
    projection = np.zeros(n)
    picked = [picked_list[0][0]]
    for i in range(1, n):
        chosen_projection = projection_list[i % number_of_projections]
        projection[i] = projection[i-1] + (chosen_projection[i] - chosen_projection[i-1])
        picked.append(picked_list[i % number_of_projections][i])
    return projection, picked

# Creates the projection of a policy which weighs multiple different factors (objectives)
# and greedily chooses zips based on the weighted total of proportions to national avg. 
def create_weighted_proj(combined_df, n=1000, objectives=['carbon_offset_metric_tons_per_panel'], weights=[1], metric='carbon_offset_metric_tons_per_panel'):

    new_df = combined_df
    new_df['weighted_combo_metric'] = combined_df[objectives[0]] * 0

    for weight, obj in zip(weights,objectives):
        new_df['weighted_combo_metric'] = new_df['weighted_combo_metric'] + (combined_df[obj] / np.mean(combined_df[obj])) * weight

    return create_greedy_projection(combined_df=new_df, n=n, sort_by='weighted_combo_metric', metric=metric)

#NEAT projection
def create_neat_projection(combined_df, state_df, n=1000, metric='carbon_offset_metric_tons_per_panel', network=None, record=True):
    #run the trained NN from model path
    data_manager = DataManager(combined_df, state_df)

    zip_outputs = [] #tuple pairs of zipcode index and score
    for i in range(0, data_manager.num_zips):
        score = network.activate(data_manager.network_inputs(i, train=False))
        zip_outputs.append((i, score))

    zip_outputs.sort(key=lambda z: z[1], reverse=True) #sort zip codes by highest score
    zip_order = [index for index, score in zip_outputs]

    #TEMP: print out the projection buckets for race
    # q = np.linspace(0, 1, 2+1)
    # thresholds = data_manager.normalized_df["black_prop"].quantile(q).to_numpy()
    # buckets, _ = data_manager.greedy_projection_bucket(zip_order, "black_prop", n=0, record=True, thresholds=thresholds)
    # print("race buckets: 0",buckets[1]-buckets[0])
    # print(buckets[0], buckets[1])

    # buckets, picked100k = data_manager.greedy_projection_bucket(zip_order, "black_prop", n=100000, record=True, thresholds=thresholds)
    # print("race buckets: 100k",buckets[1]-buckets[0])
    # print(Counter(picked100k))

    #project using original data
    projection = np.zeros(n+1)
    ordered_index = 0
    greedy_index = zip_order[ordered_index]
    existing_count = combined_df['existing_installs_count'][greedy_index]
    i = 0 #panel counter

    picked = []
    if record:
        picked = [combined_df['region_name'][greedy_index]]

    while (i < n):
        if existing_count >= combined_df['count_qualified'][greedy_index]: # this location is full
            ordered_index += 1
            greedy_index = zip_order[ordered_index]

            existing_count = combined_df['existing_installs_count'][greedy_index]

        else:
            projection[i+1] = projection[i] + combined_df[metric][greedy_index] #add a new panel to this location
            existing_count += 1
            i += 1
            if record:
                picked.append(combined_df['region_name'][greedy_index]) #record every panel location in order
    
    return projection, picked, zip_outputs


# Creates a projection of the carbon offset if we place panels to normalize the panel utilization along the given "demographic"
# I.e. if we no correlation between the demographic and the panel utilization and only fous on that, how Carbon would we offset
# TODO
def create_pop_demo_normalizing_projection(combined_df, n=1000, demographic="black_prop", metric='carbon_offset_metric_tons_per_panel'):
    pass

# Creates a projection of carbon offset for adding solar panels to random zipcodes
# The zipcode is randomly chosen for each panel, up to n panels
def create_random_proj(combined_df, n=1000, metric='carbon_offset_metric_tons_per_panel'):
    projection = np.zeros(n+1)
    picks = np.random.randint(0, len(combined_df['region_name']) -1, (n))
    for i, pick in enumerate(picks):

        while math.isnan(combined_df[metric][pick]):
            pick = np.random.randint(0, len(combined_df[metric]))
        projection[i+1] = projection[i] + combined_df[metric][pick]

    return projection

# Creates multiple different projections and returns them
def create_projections(combined_df, state_df, n=1000, load=False, metric='carbon_offset_metric_tons_per_panel', save=True, save_label=None):
    ## TODO remove rrtest (just for a new version of round robin)
    label_suffix = ""
    if save_label:
        label_suffix = "_"+save_label

    if load and exists("Clean_Data/projections_"+metric+label_suffix+".csv") and exists("Clean_Data/projections_picked"+label_suffix+".csv"):
        return pd.read_csv("Clean_Data/projections_"+metric+label_suffix+".csv"), pd.read_csv("Clean_Data/projections_picked"+label_suffix+".csv")
    
    picked = pd.DataFrame()
    proj = pd.DataFrame()
    print("Creating Continued Projection")
    proj['Status-Quo'] = create_continued_projection(combined_df, n, metric)
    print("Creating Greedy Carbon Offset Projection")
    proj['Carbon-Efficient'], picked['Carbon-Efficient'] = create_greedy_projection(combined_df, n, sort_by='carbon_offset_metric_tons_per_panel', metric=metric)
    print("Creating Greedy Average Sun Projection")
    proj['Energy-Efficient'], picked['Energy-Efficient'] = create_greedy_projection(combined_df, n, sort_by='yearly_sunlight_kwh_kw_threshold_avg', metric=metric)
    print("Creating Greedy Black Proportion Projection")
    proj['Racial-Equity-Aware'], picked['Racial-Equity-Aware'] = create_greedy_projection(combined_df, n, sort_by='black_prop', metric=metric)
    print("Creating Greedy Low Median Income Projection")
    proj['Income-Equity-Aware'], picked['Income-Equity-Aware'] = create_greedy_projection(combined_df, n, sort_by='Median_income', ascending=True, metric=metric)

    print("Creating Round Robin Projection")
    proj['Round Robin'], picked['Round Robin'] = create_round_robin_projection(projection_list=
                                                                                                   [proj['Carbon-Efficient'], proj['Energy-Efficient'], proj['Racial-Equity-Aware'], proj['Income-Equity-Aware']],
                                                                                                   picked_list=
                                                                                                   [picked['Carbon-Efficient'], picked['Energy-Efficient'], picked['Racial-Equity-Aware'], picked['Income-Equity-Aware']])
    
    print("Creating Weighted Greedy Projection")
    proj['Linear Weighted'], picked['Linear Weighted'] = create_weighted_proj(combined_df, n, ['carbon_offset_metric_tons_per_panel', 'yearly_sunlight_kwh_kw_threshold_avg', 'black_prop', 'Median_income'], [0,0.8,0.8,0.8], metric=metric)

    print("Creating NEAT Projection")

    #load model - NEAT projection moved to create_projection_models
    # with open(model_path, 'rb') as f:
    #     network = pickle.load(f)
    # proj['NEAT-Evaluation'], picked['NEAT-Evaluation'] = create_neat_projection(combined_df, state_df, n, metric=metric, network=network)

    # TESTING
    # print("Creating Random Projection")
    # proj['Random'] = create_random_proj(combined_df,n, metric)
    # print("Creating Weighted Greedy Projection")
    # proj['Weighted Greedy'], picked['Weighted Greedy'] = create_weighted_proj(combined_df, n, ['carbon_offset_metric_tons_per_panel', 'yearly_sunlight_kwh_kw_threshold_avg', 'black_prop'], [2,4,1], metric=metric)

    # uniform_samples = 10

    # print("Creating uniform random projection with", uniform_samples, "samples")

    # proj['Uniform Random (' + str(uniform_samples) + ' samples)' ] = np.zeros(n+1)
    # for i in range(uniform_samples):
    #     proj['Uniform Random (' + str(uniform_samples) + ' samples)' ] += create_random_proj(combined_df, n)/uniform_samples
    
    ## TODO remove rrtest (just for a new version of round robin)
    if save:
        proj.to_csv("Clean_Data/projections_"+metric+label_suffix+".csv",index=False)
        picked.to_csv("Clean_Data/projections_picked"+label_suffix+".csv", index=False)

    return proj, picked

def create_projections_models(combined_df, state_df, n=1000, load=False, metric='carbon_offset_metric_tons_per_panel', save=True, model_paths=["Neat/models/NEAT_model.pkl"], key_names = ["Neat"], save_label=None):
    ## TODO remove rrtest (just for a new version of round robin)
    label_suffix = ""
    if save_label:
        label_suffix = "_"+save_label

    if load and exists("Clean_Data/projections_"+metric+label_suffix+".csv") and exists("Clean_Data/projections_picked"+label_suffix+".csv") and exists("Clean_Data/projections_zip_outputs"+label_suffix+".csv"):
        #turn all values into the proper tuples
        zip_outputs = pd.read_csv("Clean_Data/projections_zip_outputs"+label_suffix+".csv", converters=
                                  {col: eval for col in range(len(pd.read_csv("Clean_Data/projections_zip_outputs"+label_suffix+".csv", nrows=0).columns))})

        return pd.read_csv("Clean_Data/projections_"+metric+label_suffix+".csv"), pd.read_csv("Clean_Data/projections_picked"+label_suffix+".csv"), zip_outputs
    
    picked = pd.DataFrame()
    proj = pd.DataFrame()
    zip_outputs = pd.DataFrame()
    print("Creating Continued Projection")
    proj['Status-Quo'] = create_continued_projection(combined_df, n, metric)

    #load models
    for model_path, key_name in zip(model_paths, key_names):
        print(f"Creating {key_name} model projection from {model_path}")
        with open(model_path, 'rb') as f:
            network = pickle.load(f)
        proj[key_name], picked[key_name], zip_outputs[key_name] = create_neat_projection(combined_df, state_df, n, metric=metric, network=network)

    if save:
        proj.to_csv("Clean_Data/projections_"+metric+label_suffix+".csv",index=False)
        picked.to_csv("Clean_Data/projections_picked"+label_suffix+".csv", index=False)
        zip_outputs.to_csv("Clean_Data/projections_zip_outputs"+label_suffix+".csv", index=False)

    return proj, picked, zip_outputs

# Creates multiple different equity projections and returns them TODO
def create_equity_projections(combined_df, picked, n=1000, load=False, metric='carbon_offset_metric_tons_per_panel', save=True, save_label=None):
    label_suffix = ""
    if save_label:
        label_suffix = "_"+save_label
        
    if load and exists("Clean_Data/equity_projections_"+metric+label_suffix+".csv"):
        return pd.read_csv("Clean_Data/equity_projections_"+metric+label_suffix+".csv")

    proj = pd.DataFrame()
    print("Creating Continued Equity Projection")
    proj['Status-Quo'] = create_continued_equity_projection(combined_df, n, metric=metric)

    #create a projection for every key in picked
    keys = picked.columns.tolist()
    for key in keys:
        if key == 'Status-Quo': #skip status quo
            continue
        print(f"Creating {key} Equity Projection")
        proj[key] = create_equity_projection_from_picked(combined_df, picked[key], n, metric=metric)

    if save:
        proj.to_csv("Clean_Data/equity_projections_"+metric+label_suffix+".csv",index=False)

    return proj


# def create_equity_projections(combined_df, picked, n=1000, load=False, metric='carbon_offset_metric_tons_per_panel', save=True, save_label=None):
#     label_suffix = ""
#     if save_label:
#         label_suffix = "_"+save_label
        
#     if load and exists("Clean_Data/equity_projections_"+metric+label_suffix+".csv"):
#         return pd.read_csv("Clean_Data/equity_projections_"+metric+label_suffix+".csv")

#     proj = pd.DataFrame()
#     print("Creating Continued Equity Projection")
#     proj['Status-Quo'] = create_continued_equity_projection(combined_df, n, metric=metric)
#     print("Creating Greedy Carbon Offset Equity Projection")
#     proj['Carbon-Efficient'] = create_equity_projection_from_picked(combined_df, picked['Carbon-Efficient'], n, metric=metric)
#     print("Creating Greedy Average Sun Equity Projection")
#     proj['Energy-Efficient'] = create_equity_projection_from_picked(combined_df, picked['Energy-Efficient'], n, metric=metric)
#     print("Creating Greedy Black Proportion Equity Projection")
#     proj['Racial-Equity-Aware'] = create_equity_projection_from_picked(combined_df, picked['Racial-Equity-Aware'], n, metric=metric)
#     print("Creating Greedy Low Median Income Equity Projection")
#     proj['Income-Equity-Aware'] = create_equity_projection_from_picked(combined_df, picked['Income-Equity-Aware'], n, metric=metric)

#     print("Creating Round Robin Equity Projection")
#     proj['Round Robin'] = create_equity_projection_from_picked(combined_df, picked['Round Robin'], n, metric=metric)

#     print("Creating NEAT Equity Projection")
#     proj['NEAT-Evaluation'] = create_equity_projection_from_picked(combined_df, picked['NEAT-Evaluation'], n, metric=metric)

#     # TESTING
#     # print("Creating Random Projection")
#     # proj['Random'] = create_random_proj(combined_df,n, metric)
#     # print("Creating Weighted Greedy Projection")
#     # proj['Weighted Greedy'], picked['Weighted Greedy'] = create_weighted_proj(combined_df, n, ['carbon_offset_metric_tons_per_panel', 'yearly_sunlight_kwh_kw_threshold_avg', 'black_prop'], [2,4,1], metric=metric)

#     # uniform_samples = 10

#     # print("Creating uniform random projection with", uniform_samples, "samples")

#     # proj['Uniform Random (' + str(uniform_samples) + ' samples)' ] = np.zeros(n+1)
#     # for i in range(uniform_samples):
#     #     proj['Uniform Random (' + str(uniform_samples) + ' samples)' ] += create_random_proj(combined_df, n)/uniform_samples
    
#     ## TODO remove rrtest (just for a new version of round robin)
#     if save:
#         proj.to_csv("Clean_Data/equity_projections_"+metric+label_suffix+".csv",index=False)

#     return proj

# Searches over many different weight settings, with the first weight being set permenantly to 1 and the other two being set proportionally
# Returns a 2d array of projections (i.e. 3d array)
def create_many_weighted(combined_df, n=1000, objectives=['carbon_offset_metric_tons_per_panel'], weight_starts=[], weight_ends=[], number_of_samples=1, metric='carbon_offset_metric_tons_per_panel', save=None, load=None):

    if exists(load):
       return np.load(load)

    all_projections = np.zeros((number_of_samples,number_of_samples,n+1))

    for i, weight1 in enumerate(np.arange(weight_starts[0], weight_ends[0], (weight_ends[0] - weight_starts[0]) / number_of_samples)):
        for j, weight2 in enumerate(np.arange(weight_starts[1], weight_ends[1], (weight_ends[1] - weight_starts[1]) / number_of_samples)):

            print("weighted proj number:", (i*number_of_samples + j))
            
            all_projections[i][j],_ = create_weighted_proj(combined_df, n=n, objectives=objectives, weights=[1, weight1, weight2], metric=metric)
    

    if save is not None:
        np.save(save, all_projections)

    return all_projections