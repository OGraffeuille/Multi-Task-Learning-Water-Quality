import sys
import os
import time
import itertools
import copy

import pandas as pd
import numpy as np


# Takes a dictionary of parameters.
# List parameters are treated as a list of parameter values to iterate through.
# Creates an iterator that goes through every parameter combination.
class param_iterator():

    def __init__(self, *params):

        # Array of parameter dicts
        self.params = [*params]

        self.iter_param_keys = []
        self.iter_param_vals = []

        self.num_combinations = 1

        # For each param in each params dict given,
        for params in self.params:
            for key in list(params):

                # if the param is in the form of a list then iterate through it
                if isinstance(params[key], list):
                    if len(params[key]) == 1:
                        params[key] = params[key][0]
                    elif len(params[key]) > 1:
                        self.iter_param_keys.append(key)
                        self.iter_param_vals.append(params[key])
                        self.num_combinations *= len(params[key])


        # Rotate through seed last, so that if we stop experiments early, we have some seed runs for all params
        if "seed" in self.iter_param_keys:
            seed_val = self.iter_param_vals[self.iter_param_keys.index("seed")]
            self.iter_param_keys.remove("seed")
            self.iter_param_keys.insert(0, "seed")
            self.iter_param_vals.remove(seed_val)
            self.iter_param_vals.insert(0, seed_val)

        # Check no two parameters have same name (would cause errors)
        assert len(self.iter_param_keys) == len(set(self.iter_param_keys)), "ERROR: Multiple iterative parameters with same name"

        # Generate all permutations of iterable parameters
        self.param_combinations = itertools.product(*self.iter_param_vals)

        # Prepare to iterate through params
        self.t0 = time.time()
        self.param_combinations_ind = 0

        print("There are {} parameter combinations to iterate through.".format(self.num_combinations))
        for i in range(len(self.iter_param_keys)):
            if len(self.iter_param_vals[i]) > 1:
                print(" - {} - {}".format(self.iter_param_keys[i], self.iter_param_vals[i]))


    def next(self):

        self.param_combinations_ind += 1
        t = time.time() - self.t0
        print(" ## NEW PARAMETER COMBINATION {}/{} ({:.0f}s)".format(self.param_combinations_ind, self.num_combinations, t))

        params_to_return = copy.deepcopy(self.params)
        iter_param_vals = next(self.param_combinations)
        for i in range(len(iter_param_vals)):
            key = self.iter_param_keys[i]
            val = iter_param_vals[i]
            for p in params_to_return:
                if key in p:
                    p[key] = val

            print(" ##  {:20} = {}".format(key, val))

        print(params_to_return)
        if len(params_to_return) == 1:
            return params_to_return[0]
        else:
            return (params for params in params_to_return)
