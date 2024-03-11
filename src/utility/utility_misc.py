import sys, time
import itertools
import copy

import numpy as np
import pandas as pd
import torch

from sklearn.utils import shuffle

# General
def display_full(item):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(item)

def count_nans(arr, msg=""):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    shp = arr.shape
    count = arr.size
    nan_count = np.count_nonzero(np.isnan(arr))
    inf_count = np.count_nonzero(np.isinf(arr))
    neginf_count = np.count_nonzero(np.isinf(arr) & np.signbit(arr))
    big_count = np.count_nonzero(arr > 1e3)
    smol_count = np.count_nonzero(arr < -1e3)
    shp = arr.shape
    print(f"{msg} - shape: {shp}, size: {count}, nan_count: {nan_count}, inf_count {inf_count}, neginf_count {neginf_count}, big_count {big_count}, smol_count {smol_count}, shape {shp}")

def check_memory_objects():
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
    lst = sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
    print(lst)

# Pytorch
def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]
    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "/n".join(lines)


# Does a few generic processing steps before saving data+metadata dfs in correct format
# data_df must have a "task" column and metadata_df a "_task" column which are used to determine task matches
# Filters for erronous (NaN) data and/or metadata, all-zero columns,
# Removes tasks w few instances
# Re-computes "task", "task_count" variables
# Saves dataframes
def save_dfs(data_df, metadata_df, fn_root, min_instances_per_task=5):

    # Merge dataframes
    join_df = pd.merge(data_df, metadata_df, left_on="task", right_on="_task")
    join_df = join_df.drop(columns=["task"])

    # Filter instances for problems
    if join_df.isna().any().any():
        print("WARNING: Removing NaN rows of data and/or metadata.")
    join_df = join_df[~join_df.isna().any(axis=1)]

    # Fix task-related meta-variables
    join_df["_task_count"] = join_df.groupby("_task").transform("count").iloc[:,0]
    join_df = join_df[join_df["_task_count"] >= min_instances_per_task]

    join_df = join_df.sort_values(by="_task_count", ascending=False)
    join_df["_task"] = join_df.groupby("_task", sort=False).ngroup()
    join_df["task"] = join_df["_task"]
    join_df = join_df.sort_values(by="_task")

    # Filter features where all vals are equal
    join_df = join_df[[c for c in join_df.columns if (len(set(join_df[c])) > 1 or c.startswith("_"))]]

    # Data df
    data_cols = [c for c in data_df.columns if c in join_df.columns]
    if "task" not in data_cols: data_cols += ["task"]
    data_df = join_df[data_cols]

    # Metadata df
    metadata_cols = [c for c in metadata_df.columns if c in join_df.columns]
    if "_task" not in metadata_cols: metadata_cols += ["_task"]
    if "_task_count" not in metadata_cols: metadata_cols += ["_task_count"]
    metadata_df = join_df[metadata_cols].drop_duplicates()
    if "_task_name" not in metadata_df.columns: metadata_df["_task_name"] = metadata_df["_task"]

    # Joined (STL) df
    join_cols = [c for c in join_df.columns if not c.startswith("_")]
    join_df = join_df[join_cols]

    # 1hot encoded STL df
    data_df_1H = data_df.copy()
    data_df_1H["t"] = data_df_1H["task"].astype(object)
    onehot = pd.get_dummies(data_df_1H[["t"]])
    data_df_1H[onehot.columns] = onehot
    data_df_1H = data_df_1H.drop("t", axis=1)

    # Save dataframes
    data_df.to_csv(fn_root+"_data.csv", index=False)
    metadata_df.to_csv(fn_root+"_metadata.csv", index=False)
    join_df.to_csv(fn_root+"_STL_metadata.csv", index=False)
    data_df_1H.to_csv(fn_root+"_STL_onehot.csv", index=False)
