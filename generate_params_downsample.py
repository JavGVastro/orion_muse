# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 12:33:06 2025

@author: ZAINTEL2
"""

# generate_params_downsample.py

import json

# Base name of the spectral line
base_name = "Ar_III-7136"

# Parameter values to explore
bins = [1,2,3,4]
flux_thresholds = [0.001]
sigma = 3.0

# Dictionary to hold all combinations
params = {}

for b in bins:
    for i, flux in enumerate(flux_thresholds):
        key = f"{base_name}_mask_bin_{2**b}"
        params[key] = {
            "bins": b,
            "flux_thresh": flux,
            "sigma_thresh": sigma
        }


#print(params)

with open("params_downsample_" + base_name + ".json", "w") as f:
    json.dump(params, f, indent=2)
