import math
import numpy as np
import torch
from plot_util import *
from data_load_util import *
from projections_util import *
from genetic_model import *
from evaluation_util import *


# try with combined_df and state_df; may get different/interesting results
combined_df = make_dataset(remove_outliers=True)
state_df = load_state_data(combined_df, load="Clean_Data/data_by_state.csv")
# print(state_df['count_qualified'].describe())
# print(combined_df['count_qualified'].describe())

def create_genetic_proj(combined_df, n=1000, metric='carbon_offset_metric_tons_per_panel'):
    projection = np.zeros(n+1)
    picks = np.random.randint(0, len(combined_df['region_name']) -1, (n))
    for i, pick in enumerate(picks):
        while math.isnan(combined_df[metric][pick]):
            pick = np.random.randint(0, len(combined_df[metric]))
        projection[i+1] = projection[i] + combined_df[metric][pick]

    return projection


# Model
res = train(combined_df, num_panels=10, num_models=100, generations=10, survivor_fraction=0.2)

torch.save(res.state_dict(), 'model.pth')

