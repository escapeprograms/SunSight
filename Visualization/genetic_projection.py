import math
import numpy as np
import torch
from plot_util import *
from data_load_util import *
from projections_util import *
from genetic_model import *
from Neat.evaluation_util import *


# try with combined_df and state_df; may get different/interesting results
combined_df = make_dataset(remove_outliers=True)
state_df = load_state_data(combined_df, load="Clean_Data/data_by_state.csv")
print(state_df['existing_installs_count'])
# print("number of states", len(state_df['State']))
# print("number of zips", len(combined_df['region_name']))
'''
State Columns
['State', 'State code', 'Clean', 'Bioenergy', 'Coal', 'Gas', 'Fossil',
       'Solar', 'Hydro', 'Nuclear', 'Wind', 'Other Renewables', 'Other Fossil',
       'Total Generation', 'Clean_prop', 'Bioenergy_prop', 'Coal_prop',
       'Gas_prop', 'Fossil_prop', 'Solar_prop', 'Hydro_prop', 'Nuclear_prop',
       'Wind_prop', 'Other Renewables_prop', 'Other Fossil_prop',
       'Total Generation_prop', 'Democrat', 'Republican', 'Total',
       'Democrat_prop', 'Republican_prop', 'Total_Population',
       'total_households', 'Median_income', 'per_capita_income',
       'households_below_poverty_line', 'black_population', 'white_population',
       'asian_population', 'native_population', 'black_prop', 'white_prop',
       'asian_prop', 'yearly_sunlight_kwh_kw_threshold_avg',
       'existing_installs_count', 'carbon_offset_metric_tons',
       'carbon_offset_metric_tons_per_panel',
       'carbon_offset_metric_tons_per_capita',
       'existing_installs_count_per_capita', 'panel_utilization',
       'realized_potential_percent', 'carbon_offset_kg_per_panel']

'''
# print(combined_df.columns)
'''
Zip Columns
['region_name', 'state_name', 'yearly_sunlight_kwh_kw_threshold_avg',
       'number_of_panels_total', 'install_size_kw_buckets_json',
       'existing_installs_count', 'percent_covered',
       'carbon_offset_metric_tons', 'count_qualified', 'square_footage',
       'solar_potential', 'solar_potential_per_capita', 'Total_Population',
       'total_households', 'Median_income', 'per_capita_income',
       'households_below_poverty_line', 'black_population', 'white_population',
       'asian_population', 'native_population', 'zcta', 'Latitude',
       'Longitude', 'zip_code', 'solar_utilization', 'panel_utilization',
       'realized_potential_percent', 'energy_generation_per_panel',
       'existing_installs_count_per_capita', 'panel_util_relative',
       'carbon_offset_metric_tons_per_panel',
       'carbon_offset_metric_tons_per_capita', 'carbon_offset_kg_per_panel',
       'asian_prop', 'white_prop', 'black_prop', 'percent_below_poverty_line']
'''

def create_genetic_proj(combined_df, n=1000, metric='carbon_offset_metric_tons_per_panel'):
    projection = np.zeros(n+1)
    picks = np.random.randint(0, len(combined_df['region_name']) -1, (n))
    for i, pick in enumerate(picks):
        while math.isnan(combined_df[metric][pick]):
            pick = np.random.randint(0, len(combined_df[metric]))
        projection[i+1] = projection[i] + combined_df[metric][pick]

    return projection


# Model
# res = train(combined_df, num_panels=10, num_models=100, generations=10, survivor_fraction=0.2)

# torch.save(res.state_dict(), 'model.pth')

