from collections import Counter
from plot_util import *
from data_load_util import *
from projections_util import *
from tqdm import tqdm
from joblib import Parallel, delayed

combined_df = make_dataset(remove_outliers=True)
state_df = load_state_data(combined_df, load="Clean_Data/data_by_state.csv")

max_num_added = 1850000
existing_count = combined_df['existing_installs_count'].sum()

#projections for all baseline strategies
load = True #set this to False if using a new model
save_label = "baseline" #get different projection saves

Energy_projections, Energy_picked = create_projections(combined_df, state_df, n=max_num_added, load=load, metric='energy_generation_per_panel', save_label=save_label)
Carbon_offset_projections, Carbon_offset_picked = create_projections(combined_df, state_df, n=max_num_added, load=load, metric='carbon_offset_kg_per_panel', save_label=save_label)
Racial_equity_projections = 1-abs(create_equity_projections(combined_df, Energy_picked, n=max_num_added, load=load, metric="black_prop", save_label=save_label)/(max_num_added+existing_count))
Income_equity_projections = 1-abs(create_equity_projections(combined_df, Energy_picked, n=max_num_added, load=load, metric="Median_income", save_label=save_label)/(max_num_added+existing_count))

#projections for NEAT selection variations

#regular lexicase: "Neat/models/01-09-25/NEAT_model2M_lexicase.pkl"
#tuned lexicase: "Neat/models/01-10-25/NEAT_model2M_lexicase.pkl"
#tuned and weighted lexicase: "Neat/models/01-12-25/NEAT_model2M_lexicase_weighted_2_2.25_1.5_1.pkl"
#tuned and weighted tournament: "Neat/models/01-09-25/NEAT_model2M_tournament.pkl"

model_paths = ["Neat/models/01-09-25/NEAT_model2M_lexicase.pkl", "Neat/models/01-12-25/NEAT_model2M_lexicase_weighted_2_2.25_1.5_1.pkl", "Neat/models/01-09-25/NEAT_model2M_tournament.pkl"]
key_names = ["NEAT-Lexicase (unweighted)", "NEAT-Lexicase", "NEAT-Tournament"] #corresponding key names to the model path
NEAT_load = True #set this to False if using a new model
NEAT_save_label = "strategy_comparison" #get different projection saves

NEAT_Energy_projections, NEAT_Energy_picked, NEAT_zip_outputs = create_projections_models(combined_df, state_df, n=max_num_added, load=NEAT_load, metric='energy_generation_per_panel', model_paths=model_paths, key_names=key_names, save_label=NEAT_save_label)
NEAT_Carbon_offset_projections, NEAT_Carbon_offset_picked, _ = create_projections_models(combined_df, state_df, n=max_num_added, load=NEAT_load, metric='carbon_offset_kg_per_panel', model_paths=model_paths, key_names=key_names, save_label=NEAT_save_label)
NEAT_Racial_equity_projections = 1-abs(create_equity_projections(combined_df, NEAT_Energy_picked, n=max_num_added, load=NEAT_load, metric="black_prop", save_label=NEAT_save_label)/(max_num_added+existing_count))
NEAT_Income_equity_projections = 1-abs(create_equity_projections(combined_df, NEAT_Energy_picked, n=max_num_added, load=NEAT_load, metric="Median_income", save_label=NEAT_save_label)/(max_num_added+existing_count))

#add all NEAT projections to baseline df
for strategy in NEAT_Energy_projections.columns:
    Energy_projections[strategy] = NEAT_Energy_projections[strategy]
    Carbon_offset_projections[strategy] = NEAT_Carbon_offset_projections[strategy]
    Racial_equity_projections[strategy] = NEAT_Racial_equity_projections[strategy]
    Income_equity_projections[strategy] = NEAT_Income_equity_projections[strategy]




print("projection calculations are done")

panel_estimations_by_year = [("Net-Zero" , 479000 * 3), ("  2030  ", 479000 * 1), ("  2034  ", 479000 * 2)]

def plot_projections(projections, panel_estimations=None, net_zero_horizontal=False, interval=1, fontsize=30, fmts=["-X", "-H", "o-", "D-", "v-", "-8", "-p"], upper_bound='Greedy Carbon Offset', ylabel=None):

    plt.style.use("seaborn")
    font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : fontsize}

    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(10, 10))

    if net_zero_horizontal:
        two_mill_continued = np.array(projections['Status-Quo'])[479000 * 3]

    keys = projections.keys()
    x = np.arange((len(projections[keys[0]]) // interval) + 1) * interval

    #scale additional panels by 1/1e6
    if panel_estimations is not None:
        for label, value in panel_estimations:
            plt.vlines(value/1e6, np.array(projections[upper_bound])[-1]/18, np.array(projections[upper_bound])[-1], colors='darkgray' , linestyles='dashed', linewidth=2, alpha=0.7)
            plt.text((value - len(projections[upper_bound])/23)/1e6, np.array(projections[upper_bound])[-1]/80, label, alpha=0.7, fontsize=25)

    if net_zero_horizontal:
        plt.hlines(two_mill_continued, 0, len(projections[upper_bound])/1e6, colors='black' , linestyles='dashed', linewidth=2, alpha=0.5)
        plt.text(0, two_mill_continued*0.85, "Continued trend at\nNet-zero prediction", alpha=0.95, fontsize=18, color='black')

    for key,fmt in zip(keys,fmts):
        plt.plot(x/1e6, np.array(projections[key])[0::interval], fmt, label=key, linewidth=3, markersize=8, alpha=0.9)


    plt.locator_params(axis='x', nbins=8) 
    plt.locator_params(axis='y', nbins=8) 
    plt.yticks(fontsize=fontsize/(2))
    plt.xticks(fontsize=fontsize/(2))

    # print("percent difference between continued and Carbon-efficient:", projections['Round Robin'].values[-1] / projections['Carbon-Efficient'].values[-1] )
    # print("percent difference between continued and racially-aware:", projections['Racial-Equity-Aware'].values[-1] / projections['Status-Quo'].values[-1])
    # print("percent difference between continued and NEAT-Evaluation:", projections['NEAT-Evaluation'].values[-1] / projections['Status-Quo'].values[-1] )

    for strategy in ['Round Robin', 'NEAT-Lexicase', 'NEAT-Tournament']:
        for i, elem in enumerate(projections[strategy].values):
            if elem > two_mill_continued:
                print(f"number of panels for net zero {strategy}:", i)
                print("percentage relative to status-quo:", i/(479000 * 3))
                break

    

    plt.xlabel("Additional Panels Built (Millions)", fontsize=fontsize, labelpad=20)
    plt.ylabel(ylabel, fontsize=fontsize, labelpad=20)
    plt.legend(fontsize=fontsize/1.7)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
    #       ncol=1, shadow=True, fontsize=fontsize/1.7)
    plt.tight_layout()
    plt.show()

# Plots a map of where the zip_codes picked are located
def plot_picked(combined_df, picked, metric, title=""):

    if metric is None:
        region_list = list(combined_df['region_name'])
        occurence_counts = picked.value_counts()
        times_picked = np.zeros_like(combined_df['region_name'])
        for pick in picked.unique():
            times_picked[region_list.index(pick)] += occurence_counts[pick]
        
        combined_df['times_picked'] = times_picked
        metric ='times_picked'

        dot_size_scale = (40 * times_picked[combined_df['times_picked']>0]/ (np.max(combined_df['times_picked'][combined_df['times_picked']>0]))) + 40     
    picked = picked.astype(str)

    geo_plot(combined_df['times_picked'][combined_df['times_picked']>0], color_scale='agsunset', title=title, edf=combined_df[combined_df['times_picked']>0], zipcodes=picked.unique(), colorbar_label="", size=dot_size_scale)

# Creates a DF with updated values of existing installs, carbon offset potential(along with per panel), and realized potential
# After a set of picks (zip codes with a panel placed in them)
def df_with_updated_picks(combined_df, picks, load=None, save=None):

    if load is not None and exists(load):
        return pd.read_csv(load)

    new_df = combined_df
    new_co = np.array(new_df['carbon_offset_metric_tons'])
    new_existing = np.array(new_df['existing_installs_count'])

    for pick in tqdm(picks):
        index = list(new_df['region_name']).index(pick)
        new_co[index] -= new_df['carbon_offset_metric_tons_per_panel'][index]
        new_existing[index] += 1
    
    print('carbon offset difference:', np.sum(new_df['carbon_offset_metric_tons'] - new_co))
    new_df['carbon_offset_metric_tons'] = new_co
    new_df['carbon_offset_kg'] = new_co * 1000
    print('Number install change:', np.sum(new_existing - new_df['existing_installs_count']) )
    new_df['existing_installs_count'] = new_existing
    new_df['existing_installs_count_per_capita'] = new_existing / new_df['Total_Population']
    new_df['panel_utilization'] = new_existing / new_df['number_of_panels_total']

    if save is not None:
        new_df.to_csv(save, index=False)

    return new_df

def plot_demo_state_stats(new_df,save="Clean_Data/data_by_state_proj.csv"):
    state_df = load_state_data(new_df, load=None, save=save)

    hatches=['o','o','o','o','o','x','x','x','x','x']
    annotate = False
    type = 'paper'
    stacked = False

    bar_plot_demo_split(state_df, demos=["black_prop", "white_prop","Median_income", "asian_prop", "Republican_prop"], key="panel_utilization", xticks=['Black', 'White', 'Asian','Income','Republican'] , type=type, stacked=stacked, ylabel="Realized Potential (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True)
    bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="existing_installs_count_per_capita", type=type, stacked=stacked, ylabel="Existing Installs Per Capita (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True)
    bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="carbon_offset_kg", type=type, stacked=stacked, ylabel="Carbon Offset Potential (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True) 

    hatches=['o','o','o','o','x','x','x','x']

    bar_plot_demo_split(new_df, demos=["black_prop", "white_prop","Median_income", "asian_prop"], key="panel_utilization", xticks=['Black', 'White', 'Asian','Income'] , type=type, stacked=stacked, ylabel="Realized Potential (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True) 
    bar_plot_demo_split(new_df, demos=["black_prop", "white_prop","Median_income", "asian_prop"], key="carbon_offset_kg", xticks=['Black', 'White', 'Asian','Income'] , type=type, stacked=stacked, ylabel="Potential Carbon Offset (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True)
    bar_plot_demo_split(new_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop"], xticks=['Black', 'White', 'Asian', 'Income'], key="existing_installs_count_per_capita", type=type, stacked=stacked, ylabel="Existing Installs Per Capita (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True)

# plot_projections(Carbon_offset_projections, panel_estimations_by_year, net_zero_horizontal=True, interval=100000, upper_bound='Carbon-Efficient', ylabel="Carbon Offset (kg)")
# plot_projections(Energy_projections, panel_estimations_by_year, net_zero_horizontal=True, interval=100000, upper_bound='Energy-Efficient', ylabel="Additional Energy Capacity (kWh)")

# print(Energy_picked[''])

# for key in ['Energy-Efficient', 'Carbon-Efficient', 'Racial-Equity-Aware', 'Income-Equity-Aware', 'Round Robin']:
#     plot_picked(combined_df, Energy_picked[key], None, title="")

#print NEAT picks
# print(Energy_picked['NEAT-Evaluation'].head())
# plot_picked(combined_df, Energy_picked['NEAT-Evaluation'], None, title="")
# quit()

def weighted_proj_heatmap(combined_df, metric='carbon_offset_kg_per_panel', objectives=['carbon_offset_kg_per_panel', 'energy_generation_per_panel', 'black_prop']):
    weight_starts = [0.0, 0.0]
    weight_ends = [0.5,1.5]
    number_of_samples = 5
    weighted_proj_array = create_many_weighted(combined_df, n=1850000, objectives=objectives, weight_starts=weight_starts, weight_ends=weight_ends, number_of_samples=5, metric=metric,
                                               save='Projection_Data/weighted_map_5_energy', load='Projection_Data/weighted_map_5_energy.npy')


    ax = sns.heatmap(weighted_proj_array[:,:,-1], xticklabels=np.round(np.arange(weight_starts[0],weight_ends[0], (weight_ends[0] - weight_starts[0])/number_of_samples), 1), yticklabels=np.round(np.arange(weight_starts[1],weight_ends[1], (weight_ends[1] - weight_starts[1])/number_of_samples), 1))
    ax.set_xlabel("Energy Potential Weight")
    ax.set_ylabel("Black Prop Weight")
    plt.show()


# weighted_proj_heatmap(combined_df, metric='energy_generation_per_panel')

# quit()

# co_df = df_with_updated_picks(combined_df, Energy_picked['Greedy Carbon Offset'], load='Projection_Data/df_greedy_co.csv', save='Projection_Data/df_greedy_co.csv')
# round_robin_df = df_with_updated_picks(combined_df, Energy_picked['Round Robin Policy'], load='Projection_Data/df_greedy_rrtest_rr.csv', save='Projection_Data/df_greedy_rrtest_rr.csv')
# energy_df = df_with_updated_picks(combined_df, Energy_picked['Greedy Average Sun'], load='Projection_Data/df_greedy_sun.csv', save='Projection_Data/df_greedy_sun.csv')
# black_df = df_with_updated_picks(combined_df, Energy_picked['Greedy Black Proportion'], load='Projection_Data/df_greedy_black.csv', save='Projection_Data/df_greedy_black.csv')
# weighted_df = df_with_updated_picks(combined_df, Energy_picked['Weighted Greedy'], load='Projection_Data/df_greedy_weighted.csv', save='Projection_Data/df_greedy_weighted.csv')

# plot_demo_state_stats(round_robin_df, save="Projection_Data/data_by_state_proj_greedy_round_robink.csv")
# plot_demo_state_stats(energy_df, save="Projection_Data/data_by_state_proj_greedy_weighted.csv")


def dominates(sol1, sol2):
    """Check if sol1 dominates sol2."""
    return all(s1 >= s2 for s1, s2 in zip(sol1, sol2)) and any(s1 > s2 for s1, s2 in zip(sol1, sol2))


# def grid_search(combined_df, npanels, metrics, objectives, save = None, load = None):
#     """Perform grid search to find Pareto-optimal solutions."""
#     weight_starts = [0.0, 0.0, 0.0, 0.0]
#     weight_ends = [2.0, 2.0, 2.0, 2.0]
#     number_of_samples = 10
#     filename = "/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/GS" + str(number_of_samples)
#     results = {}

#     for i, weight1 in enumerate(np.arange(weight_starts[0], weight_ends[0], (weight_ends[0] - weight_starts[0]) / number_of_samples)):
#         for j, weight2 in enumerate(np.arange(weight_starts[1], weight_ends[1], (weight_ends[1] - weight_starts[1]) / number_of_samples)):
#             for k, weight3 in enumerate(np.arange(weight_starts[2], weight_ends[2], (weight_ends[2] - weight_starts[2]) / number_of_samples)):
#                 for l, weight4 in enumerate(np.arange(weight_starts[3], weight_ends[3], (weight_ends[3] - weight_starts[3]) / number_of_samples)):

#                     key = ' '.join(str(x) for x in [weight1,weight2,weight3,weight4])
#                     results[key] = []

#     for metric in metrics:
#         print("current metric is" + metric)
#         for weights, projectvalue in results.items():
#             print("creating projection array for weights "+ weights)
#             weightarr = list(map(float, weights.split(" ")))
#             projection,picked = create_weighted_proj(combined_df, n=npanels, objectives=objectives, weights=weightarr, metric=metric)
#             if metric == 'black_prop' or metric == 'Median_income':
#                 new_picks_df = df_with_updated_picks(combined_df, picked)
#                 # print(new_picks_df)
#                 key = "panel_utilization"
#                 if metric == 'black_prop':
#                     demo = 'black_prop'
#                 else:
#                     demo = 'Median_income'
#                 median = np.median(new_picks_df[demo].values)
#                 low_avg = np.mean(new_picks_df[new_picks_df[demo] < median][key].values)
#                 high_avg = np.mean(new_picks_df[new_picks_df[demo] >= median][key].values)
#                 equity_score = np.abs(1-np.abs(high_avg-low_avg))
#                 projectvalue.append(equity_score)
#                 continue
#             if metric == 'carbon_offset_kg_per_panel':
#                 projectvalue.append(projection[-1]/Carbon_offset_projections['Status-Quo'].values[-1])
#                 continue
#             if metric == 'energy_generation_per_panel':
#                 projectvalue.append(projection[-1]/Energy_projections['Status-Quo'].values[-1])
#                 continue
#     data = []
#     for key, values in results.items():
#         weights = list(map(float, key.split()))
#         row = weights + values
#         data.append(row)
    
#     columns = ['w1', 'w2', 'w3', 'w4', 'CO (prop to SQ)', 'Energy gen (prop to SQ)', 'Income_EQ', 'racial_EQ']
    
#     df = pd.DataFrame(data, columns=columns)
    
#     # Save the DataFrame to CSV
#     df.to_csv(filename, index=False)

#     return df
def grid_search(combined_df, npanels, metrics, objectives, save=None, load=None):
    """Perform grid search to find Pareto-optimal solutions."""
    weight_starts = [0.0, 0.0, 0.0, 0.0]
    weight_ends = [2.0, 2.0, 2.0, 2.0]
    number_of_samples = 12
    filename = "/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/GS" + str(number_of_samples)
    
    # Precompute grid of weights
    weight_values = np.linspace(weight_starts[0], weight_ends[0], number_of_samples)
    weight_grid = np.array(np.meshgrid(weight_values, weight_values, weight_values, weight_values)).T.reshape(-1, 4)

    results = {tuple(weights): [] for weights in weight_grid}

    # Preload projections
    co_baseline = Carbon_offset_projections['Status-Quo'].values[-1]
    energy_baseline = Energy_projections['Status-Quo'].values[-1]

    def process_metric(metric):
        """Process a single metric."""
        local_results = {}
        for weights in weight_grid:
            weights_tuple = tuple(weights)
            projection, picked = create_weighted_proj(combined_df, n=npanels, objectives=objectives, weights=weights, metric=metric)
            if metric in ['black_prop', 'Median_income']:
                new_picks_df = df_with_updated_picks(combined_df, picked)
                key = "panel_utilization"
                demo = 'black_prop' if metric == 'black_prop' else 'Median_income'
                median = np.median(new_picks_df[demo].values)
                low_avg = np.mean(new_picks_df[new_picks_df[demo] < median][key].values)
                high_avg = np.mean(new_picks_df[new_picks_df[demo] >= median][key].values)
                equity_score = np.abs(1 - np.abs(high_avg - low_avg))
                local_results[weights_tuple] = equity_score
            elif metric == 'carbon_offset_kg_per_panel':
                local_results[weights_tuple] = projection[-1] / co_baseline
            elif metric == 'energy_generation_per_panel':
                local_results[weights_tuple] = projection[-1] / energy_baseline
        return metric, local_results

    # Parallelise metric computation
    metrics_results = Parallel(n_jobs=-1)(delayed(process_metric)(metric) for metric in metrics)

    # Combine results
    for metric, metric_results in metrics_results:
        for weights_tuple, value in metric_results.items():
            results[weights_tuple].append(value)

    # Create DataFrame
    data = [list(weights) + values for weights, values in results.items()]
    columns = ['w1', 'w2', 'w3', 'w4', 'CO (prop to SQ)', 'Energy gen (prop to SQ)', 'Income_EQ', 'racial_EQ']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)

    return df
def pareto_calc(df):  
    pareto_opt_solutions = []
    filename = "/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/GSParetoOptSoln12"
    for index, row in df.iterrows():
        add = True
        for i, optsoln in enumerate(pareto_opt_solutions):
            if dominates(row[4:],optsoln[4:]):
                del(pareto_opt_solutions[i])
            if dominates(optsoln, row): 
                add = False
                break
        if add == True:
            pareto_opt_solutions.append(row)
        # print(pareto_opt_solutions)
    columns = ['w1', 'w2', 'w3', 'w4', 'CO (prop to SQ)', 'Energy gen (prop to SQ)', 'Income_EQ', 'racial_EQ']
    new_df = pd.DataFrame(pareto_opt_solutions, columns=columns)
    new_df.to_csv(filename, index=False)
    return df


#TODO make compariosn_keys plural so we can plot multiple 

def plot_comparison_ratio(all_metric_projections, base_key, comparison_key, metric_labels = ['Carbon Offset', 'Energy Generation', 'Racial Equity', 'Income Equity'], interval = 1, fontsize=30, fmts=["-X", "-H", "o-", "D-", "v-", "-8", "-p"], title="Performance of NEAT"):
    
    plt.style.use("seaborn")
    font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : fontsize}

    matplotlib.rc('font', **font)


    fig, ax = plt.subplots(figsize=(10, 6))

    #calculate ratios between the base and the comparison
    ratios = pd.DataFrame()
    for metric, projection in zip(metric_labels, all_metric_projections):
        ratios[metric] = projection[comparison_key]/projection[base_key]
        print(f"{metric} ratio: {ratios[metric][len(ratios[metric_labels[0]])-1]}")

    #hard code upper bounds for greedy CO and EG
    co_id = metric_labels.index('Carbon Offset')
    eg_id = metric_labels.index('Energy Generation')
    ratios['co max'] = all_metric_projections[co_id]["Carbon-Efficient"]/all_metric_projections[co_id][base_key]
    ratios['eg max'] = all_metric_projections[eg_id]["Energy-Efficient"]/all_metric_projections[eg_id][base_key]

    #plot ratios
    x = np.arange((len(ratios[metric_labels[0]]) // interval) + 1) * interval/1000000 #scale down by 10e6

    for metric, fmt in zip(metric_labels, fmts):
        plt.plot(x, np.array(ratios[metric])[0::interval], fmt, label=metric, linewidth=3, markersize=8, alpha=0.9)

    #plot code upper bounds
    # plt.plot(x, np.array(ratios['co max'])[0::interval], "--X", color="tab:blue", label='Maximum Possible Carbon Offset', linewidth=3, markersize=8, alpha=0.9)
    # plt.plot(x, np.array(ratios['eg max'])[0::interval], "--H", color="tab:green", label='Maximum Possible Energy Generation', linewidth=3, markersize=8, alpha=0.9)

    #show baseline
    plt.hlines(1, 0, x[-1], colors='black' , linestyles='dashed', linewidth=2, alpha=0.5)

    # plt.ylim(0, 2) #set the range of the plots
    ax.tick_params(axis='both', labelsize=fontsize/2)
    plt.xlabel("Additional Panels Built (millions)", fontsize=fontsize, labelpad=20)
    plt.ylabel(f"Ratio to {base_key}", fontsize=fontsize, labelpad=20)
    plt.legend(fontsize=fontsize/1.7)
    # plt.title(title, fontsize=fontsize/1.2)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
    #       ncol=1, shadow=True, fontsize=fontsize/1.4)
    
    plt.tight_layout()
    plt.show()


#this funciton is for testing only
def plot_comparison(all_metric_projections, base_key, comparison_key, metric_labels = ['Carbon Offset', 'Energy Generation', 'Racial Equity', 'Income Equity'], interval = 1, fontsize=30, fmts=["-X", "-H", "o-", "D-", "v-", "-8", "-p"], title="Comparison", ylabel="Value"):
    
    plt.style.use("seaborn")
    font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : fontsize}

    matplotlib.rc('font', **font)

    #calculate ratios between the base and the comparison
    values = pd.DataFrame()
    for metric, projection in zip(metric_labels, all_metric_projections):
        values[metric] = projection[comparison_key]
        values[f"{metric}-base"] = projection[base_key]

    #plot ratios
    x = np.arange((len(values[metric_labels[0]]) // interval) + 1) * interval

    for metric, fmt in zip(metric_labels, fmts):
        plt.plot(x, np.array(values[metric])[0::interval], fmt, label=f"{metric} {comparison_key}", linewidth=3, markersize=8, alpha=0.9)

        plt.plot(x, np.array(values[f"{metric}-base"])[0::interval], fmt, label=f"{metric} {base_key}", linewidth=3, markersize=8, alpha=0.9)
        plt.xlabel("Additional Panels Built", fontsize=fontsize, labelpad=20)

    #show baseline
    # plt.hlines(1, 0, x[-1], colors='black' , linestyles='dashed', linewidth=2, alpha=0.5)

    # plt.ylim(0, 2) #set the range of the plots
    plt.ylabel(ylabel, fontsize=fontsize, labelpad=20)
    plt.legend(fontsize=fontsize/1.5)
    # plt.title(title, fontsize=fontsize)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
    #       ncol=1, shadow=True, fontsize=fontsize/1.4)
    plt.tight_layout()
    plt.show()

#equity check
# plot_comparison([Racial_equity_projections, Income_equity_projections], "Status-Quo", "NEAT-Evaluation", metric_labels = ['Racial Equity', 'Income Equity'], interval = 100000, title="Realized Potential Disparity for Lexicase", ylabel="Realized Potential Disparity across Median")


def plot_bar_comparison_ratio(all_metric_projections, base_key, method_keys = ["NEAT_model"], method_names = ['Lexicase'], metric_labels = ['carbon_offset', 'energy_generation', 'racial_equity', 'income_equity'], fontsize=30):
    
    plt.style.use("seaborn")
    font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : fontsize}

    matplotlib.rc('font', **font)

    #get the last value for all objectives for all methods
    results = [] #ex: array of [lexicase results, tournament results etc.]
    for key in method_keys:
        result = [] #array of [CO, EG, RE, IE]
        for projection in all_metric_projections:
            result.append(projection[key].tolist()[-1] / projection[base_key].tolist()[-1]) #get the ratio to the base key
        results.append(result)

    # Configuration for the bar graph
    x = np.arange(len(metric_labels))  # X positions for the groups
    width = 0.17  # Width of each bar

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add bars for each method
    for i, method in enumerate(method_names):
        ax.bar(x + i * width, results[i], width, label=method)
    
    #show baseline
    plt.axhline(y=1, color='black', linestyle='--', linewidth=2) 

    # Add labels, title, and legend
    ax.set_xlabel('Objectives', fontsize=fontsize, labelpad=20)
    ax.set_ylabel(f'Ratio to {base_key}', fontsize=fontsize, labelpad=20)
    # ax.set_title('Fitness of Selection Methods for all Objectives', fontsize=fontsize/1.2)
    
    ax.tick_params(axis='both', labelsize=fontsize/2)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(metric_labels, fontsize=fontsize/2.2)
    ax.legend(fontsize=fontsize/2, loc="upper center", bbox_to_anchor=(0.4, 1)) #hardcoded legend placement

    # Show the plot
    plt.tight_layout()
    plt.show()

def scored_geo_plot(combined_df, zip_outputs, title):
    zip_outputs = zip_outputs.tolist()

    #use log scaling
    zip_dict = dict([(index, np.exp(score[0])) for index, score in zip_outputs])
    combined_df['scores'] = combined_df.index.map(zip_dict)

    geo_plot(combined_df['scores'], color_scale='agsunset_r', title=None, edf=combined_df, zipcodes=combined_df, colorbar_label="", size=20)

#Strategy X vs Status Quo
#unweighted
plot_comparison_ratio([Carbon_offset_projections, Energy_projections, Racial_equity_projections, Income_equity_projections], "Status-Quo", "NEAT-Lexicase (unweighted)",
                      metric_labels = ['Carbon Offset', 'Energy Generation', 'Racial Equity', 'Income Equity'], interval = 100000, title="Performance of Unweighted NEAT-Lexicase")

# # weighted lexicase selection
# plot_comparison_ratio([Carbon_offset_projections, Energy_projections, Racial_equity_projections, Income_equity_projections], "Status-Quo", "NEAT-Lexicase",
#                       metric_labels = ['Carbon Offset', 'Energy Generation', 'Racial Equity', 'Income Equity'], interval = 100000, title="Performance of Weighted NEAT-Lexicase")

# # tournmaent selection
# plot_comparison_ratio([Carbon_offset_projections, Energy_projections, Racial_equity_projections, Income_equity_projections], "Status-Quo", "NEAT-Tournament",
#                       metric_labels = ['Carbon Offset', 'Energy Generation', 'Racial Equity', 'Income Equity'], interval = 100000, title="Performance of NEAT-Tournament")

# #Linear Weighted
# plot_comparison_ratio([Carbon_offset_projections, Energy_projections, Racial_equity_projections, Income_equity_projections], "Status-Quo", "Linear Weighted",
#                       metric_labels = ['Carbon Offset', 'Energy Generation', 'Racial Equity', 'Income Equity'], interval = 100000, title="Performance of Weighted Linear Weighted")

# #Round Robin
# plot_comparison_ratio([Carbon_offset_projections, Energy_projections, Racial_equity_projections, Income_equity_projections], "Status-Quo", "Round Robin",
#                       metric_labels = ['Carbon Offset', 'Energy Generation', 'Racial Equity', 'Income Equity'], interval = 100000, title="Performance of Weighted Linear Weighted")



#Comparison Bar Chart
all_key_names = ["NEAT-Lexicase", "NEAT-Tournament","Linear Weighted", "Round Robin"]
plot_bar_comparison_ratio([Carbon_offset_projections, Energy_projections, Racial_equity_projections, Income_equity_projections],
                          base_key="Status-Quo", method_keys=all_key_names, method_names=all_key_names,
                          metric_labels=['Carbon Offset', 'Energy Generation', 'Racial Equity', 'Income Equity'])


# #Geo plots for different methods
# print(NEAT_zip_outputs['NEAT-Tournament'].head(100))
scored_geo_plot(combined_df, NEAT_zip_outputs['NEAT-Lexicase'], "NEAT-Lexicase")
# scored_geo_plot(combined_df, NEAT_zip_outputs['NEAT-Tournament'], "NEAT-Tournament")
# scored_geo_plot(combined_df, NEAT_zip_outputs['LINEAR SCORES???'], "LINARUUU")

#Projections
Carbon_offset_projections_plot = Carbon_offset_projections[["Status-Quo","NEAT-Lexicase", "NEAT-Tournament","Linear Weighted", "Round Robin", "Carbon-Efficient"]]
plot_projections(Carbon_offset_projections_plot, panel_estimations_by_year, net_zero_horizontal=True, interval=100000, upper_bound='Carbon-Efficient', ylabel="Carbon Offset (kg)")

Energy_projections_plot = Energy_projections[["Status-Quo","NEAT-Lexicase", "NEAT-Tournament","Linear Weighted", "Round Robin", "Energy-Efficient"]]
plot_projections(Energy_projections_plot, panel_estimations_by_year, net_zero_horizontal=True, interval=100000, upper_bound='Energy-Efficient', ylabel="Additional Energy Capacity (kWh)")
