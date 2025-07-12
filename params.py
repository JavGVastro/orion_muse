# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 20:12:00 2025

@author: ZAINTEL2
"""

# params.py
import sys
import json

names       = ['H_I-6563', 'O_III-5007', 'O_III-4959', 'N_II-6548', 'N_II-6583', 'S_III-9069']
bins        = [4,          4,           4,           4,           4,           4]
flux_thresh = [0.1,        0.05,        0.0167,      0.0167,      0.05,        0.01]
sigma       = [2.0,        2.0,         2.0,         2.0,         2.0,         2.0]
data_label  = ['Orion Ha', 'Orion [OIII]', 'Orion [OIII]', 'Orion [NII]', 'Orion [NII]', 'Orion [SIII]']

data = {
    name: {
        'bins': bins[i],
        'flux': flux_thresh[i],
        'sigma': sigma[i],
        'data': data_label[i]
    }
    for i, name in enumerate(names)
}

query = sys.argv[1]
print(json.dumps(data[query]))