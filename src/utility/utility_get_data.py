import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .utility_misc import display_full

# Creates a mapping from groups to new groups, where the new groups have the following differences:
# - Groups with less than min_group_size members are mapped to 0
# - Groups with more than min_group_size members are mapped to 1,2,... keeping original order
def regroup_min_size_map(groups, min_group_size, groups_unique=None):

    if groups_unique is None:
        groups_unique = np.unique(groups)

    need_group_0 = False
    for g in groups_unique:
        if np.sum(g == groups) < min_group_size:
            need_group_0 = True
            break

    # Create default mapping
    if not need_group_0:
        group_map = {g: i for i, g in enumerate(groups_unique)}

    # Create mapping with group 0
    else:
        group_map = {}
        map_ind = 1
        g0_size = 0
        for g in groups_unique:
            if np.sum(g == groups) < min_group_size:
                group_map[g] = 0
                g0_size += np.sum(g == groups)
            else:
                group_map[g] = map_ind
                map_ind += 1

        # If the group 0 has insufficient members, then map smallest group to 0 also
        if g0_size > 0 and g0_size < min_group_size:
            for g in groups_unique:
                if group_map[g] == map_ind-1: # Last group to be mapped
                    group_map[g] = 0

    return group_map

# Multispectral conversion
# inputs
#        hyper_data - DataFrame of hyperspectral data to convert
#     hyper_lambdas - Numpy array (n): lambda of each column of hyper_data
#       multi_bands - String: name of bands to convert to (determines what response file to use)
def hyper2multi(hyper_df, hyper_lambdas, multi_bands='olci'):

    if multi_bands == 'olci':
        response = np.genfromtxt(r'C:\Users\ogra439\Documents\TAIAO\scripts\MTL\data\hyper2multi\S3A_OL_SRF_20160713_mean_rsr.csv', delimiter=',', skip_header=1)
        response_lambdas = response[:,0].astype(int)
        response_data = response[:,1:]
        multi_wavelengths = ["400", "412.5", "442.5", "490", "510", "560", "620", "665", "673.75", "681.25", "708.75", "753.75", "761.25", "764.375", "767.5", "778.75", "865", "885", "900", "940", "1020"]
    else:
        assert False, "Unrecognised multi_bands string (try 'olci')"

    # Ensure hyperspectral data and reponse data share same wavelengths
    # We assume an ordered dataset with lambda=1 intervals between data

    # Get lambdas that are in both arrays
    hyper_lambdas_keep = []
    for l in response_lambdas:
        vals = np.where(l == hyper_lambdas)[0]
        if len(vals) > 0:
            hyper_lambdas_keep.append(vals[0])

    response_lambdas_keep = []
    for l in hyper_lambdas:
        vals = np.where(l == response_lambdas)[0]
        if len(vals) > 0:
            response_lambdas_keep.append(vals[0])

    # Filter data to only keep correct lambdas
    hyper_data = hyper_df.values
    hyper_data = hyper_data[:, hyper_lambdas_keep]
    response_data = response_data[response_lambdas_keep, :]
    lambdas = hyper_lambdas[hyper_lambdas_keep]
    num_lambdas = len(lambdas)

    # Compute signal
    multi_data = np.array(np.matrix(hyper_data) @ np.matrix(response_data))
    
    # Format into DataFrame
    multi_df = pd.DataFrame(multi_data, columns=multi_wavelengths)
    all_same_cols = multi_df.columns[multi_df.nunique() == 1]
    multi_df = multi_df.drop(all_same_cols, axis=1)
    return multi_df


# Filter spectra by removing the columns outside the range
def trim_spectra(df, lambda_min, lambda_max, colname="Rrs_{}"):
    return df[[colname.format(i) for i in range(lambda_min, lambda_max+1)]]


# Normalise spectra by dividing by area 
# Justification: Optical types of inland and coastal waters
def normalise_spectra(df):
    return df.div(df.sum(axis=1), axis=0)


# Separate data into Optical Water Types
def get_OWTs(X):
    pass


# Main function that loads data from files and returns it in a format ready for experiments
def get_data(datasets, labels, **kwargs):

    if kwargs["split_method"] == "test_on_LOO_dataset" and "LOO_dataset" in kwargs and kwargs["LOO_dataset"] is not None:
        print("WARNING: Only including {} in test sets.".format(datasets[kwargs["LOO_dataset"]]))

    if "matchup" in datasets and len(datasets) > 1: 
        NotImplementedError("Matchup data must be used alone.")

    if isinstance(datasets, str):
        datasets = [datasets]

    # Separate dataset_specific arguments with generic arguments
    dataset_kwargs = {}
    for dataset in datasets:
        dataset_kwargs[dataset] = {}
        for key in kwargs:
            if key.startswith(dataset+"_"):
                dataset_kwargs[dataset][key[len(dataset)+1:]] = kwargs[key]
    kwargs = {key: kwargs[key] for key in kwargs if not key.startswith(tuple(datasets))}

    # Load datasets
    dfs = []
    if "nz" in datasets:
        dfs.append(get_NZ_data(labels, **dataset_kwargs["nz"]))
    if "gloria" in datasets:
        exclude_nz_data = "nz" in datasets
        dfs.append(get_GLORIA_data(labels, exclude_nz_data=exclude_nz_data, **dataset_kwargs["gloria"]))
    elif "matchup" in datasets:
        dfs.append(get_matchup_data(labels, **dataset_kwargs["matchup"]))
    if len(dfs) != len(datasets):
        assert False, "ERROR: ({} != {}) At least one incorrect dataset: {}.".format(len(dfs), len(datasets), datasets)

    # Format data
    data = format_data(dfs, labels, **kwargs)

    return data
    
# Processes and returns NZ data
#                    labels - list of strings: labels to use (chl, tss, cdom)
#          max_matchup_diff - Int: maximum number of days between satellite and in situ data
#               site_subset - list of strings: sites to use (None for all)
#         multiple_matchups - Bool: whether to use multiple matchups per site
def get_NZ_data(labels, version=3, max_matchup_diff=None, site_subset=None, multiple_matchups=False):

    if len(labels) > 1:
        NotImplementedError("Multiple labels not implemented for NZ data.")
    filenames = {
        "chl": {
            1: r"data\NZ\s3data_match_3days.csv",
            2: r"data\NZ\s3_chl_3_day.csv",
            3: r"data\NZ\s3_chl_3_day 3.csv"
        }, "tss": {
            1: None,
            2: r"data\NZ\s3_tss_0_day.csv",
            3: r"data\NZ\s3_tss_0_day 2.csv"
        }
    }
    filename = filenames[labels[0]][version]
    
    datediff_col = 'dateDiff'
    site_col = "site"
    date_col = 'date_nzst.y'
    if version == 1:
        label_cols = {"chl": "CensoredValue"}
    elif version in [2, 3]:
        label_cols = {"chl": "chla_insitu", "tss": "tss_insitu"}
    feature_col_start = "rhow_" if version in [1, 2] else "mean.rhow_"
    
    multi_wavelengths = ["400", "412.5", "442.5", "490", "510", "560", "620", "665", "673.75", "681.25", "708.75", "753.75", "761.25", 
                         "764.375", "767.5", "778.75", "865", "885", "900", "940", "1020"]
    data_df = pd.read_csv(filename)

    ### Filter ###
    if not max_matchup_diff is None:
        data_df = data_df[data_df[datediff_col] <= max_matchup_diff] 

    if site_subset is not None:
        data_df = data_df[data_df[site_col].isin(site_subset)]

    if multiple_matchups == False:
        data_df = data_df.loc[data_df.groupby([site_col, date_col])[datediff_col].idxmin()]

    ### Reflectance ###
    rrs_colnames_old = [c for c in data_df.columns if c.startswith(feature_col_start)]
    rrs_colnames = ["feature_" + multi_wavelengths[int(c.split("_")[1].split(".")[0])-1] for c in rrs_colnames_old]
    column_mapping = {old_name: new_name for old_name, new_name in zip(rrs_colnames_old, rrs_colnames)}
    data_df.rename(columns=column_mapping, inplace=True)

    # Turn rho w into normalised r_rs
    data_df[rrs_colnames] = data_df[rrs_colnames] / np.pi
    
    ### Label(s) ###
    if "cdom" in labels:
        print("WARNING: CDOM labels not available for NZ data.")
    if isinstance(labels, str):
        labels = [labels]
    for label in labels:
        data_df[f"label_{label}"] = data_df[label_cols[label]]
    label_colnames = [f"label_{label}" for label in labels]

    ### Site ###
    data_df["site"] = data_df[site_col] 
    
    ### Merge data ###
    data_df = data_df[rrs_colnames + label_colnames + ["site"]]
    
    return data_df


# Processes and returns GLORIA data
#                    labels - list of strings: labels to use (chl, tss, cdom)
#             convert2multi - String: what multispectral bands to convert to (olci)
def get_GLORIA_data(labels, convert2multi="olci", exclude_nz_data=False, n_data=None, site_subset=None):

    filepath = "data\GLORIA"
    filename_rrs = "GLORIA_Rrs.csv"
    filename_labels_metadata = "GLORIA_meta_and_lab.csv"
    wavelength_range = [400, 800]
    wavelengths = np.arange(wavelength_range[0], wavelength_range[1]+1)
    keep_bands = [*range(1,16)] # Don't use Oa01
    
    # Check GLORIA data has been downloaded
    assert os.path.exists(os.path.join(filepath, filename_rrs)), "ERROR: GLORIA data not found at {}.".format(os.path.join(filepath, filename_rrs))

    rrs_df = pd.read_csv(os.path.join(filepath, filename_rrs))
    label_metadata_df = pd.read_csv(os.path.join(filepath, filename_labels_metadata))
    
    ### Filter ###
    if site_subset is not None:
        label_metadata_df = label_metadata_df[label_metadata_df["Site_name"].isin(site_subset)]

    if exclude_nz_data:
        label_metadata_df = label_metadata_df[label_metadata_df["Country"] != "New Zealand"]

    ### Reflectance ###
    rrs_df = trim_spectra(rrs_df, wavelength_range[0], wavelength_range[1])
    if convert2multi == "olci":
        rrs_df = hyper2multi(rrs_df, wavelengths, "olci")
    rrs_df.columns = ["feature_" + str(wavelength) for wavelength in rrs_df.columns]

    rrs_df = rrs_df[[rrs_df.columns[i] for i in keep_bands]]

    ### Label(s) ###
    label_df = pd.DataFrame()
    label_var_names = {"chl": "Chla", "tss": "TSS", "cdom": "aCDOM440"}
    if isinstance(labels, str):
        labels = [labels]
    for label in labels:
        label_df[f"label_{label}"] = label_metadata_df[label_var_names[label]]
    label_colnames = label_df.columns.tolist()
    assert len(label_colnames) > 0, "ERROR: Unknown labels: {}.".format(labels)

    ### Site ###
    label_df["site"] = label_metadata_df["Site_name"] 
    label_df["site"] = label_df["site"].str.split(",").str[0]

    ### Merge data Merge the two to keep only valid data
    rrs_df = rrs_df.dropna()
    data_df = pd.merge(rrs_df, label_df, left_index=True, right_index=True, how='inner')
    
    # Remove data if necessary
    if not n_data is None:
        data_df = data_df.sample(n=n_data, random_state=0)

    return data_df


# Combines datasets, transforms them, adds tasks, partitions, to run experiments on
#                  datasets - list of DataFrames: datasets to combine
#                    labels - list of strings: labels to use (chl, tss, cdom)
#            label_ln_coefs - Dictionary: label name to coefficient to add to label before taking log
#      invalid_label_action - String: what to do with data with invalid labels (drop_if_any, drop_if_all, impute)
#  task_selection_criterion - String: how to select tasks (TSS1, cluster, site)
# task_selection_n_clusters - (if task_selection_criterion is cluster) Int: number of clusters to use 
#             min_site_size - Int: minimum number of data points in a site to keep it 
#              split_method - String: how to split data into train, validation and test sets 
#                             (random, random_equal_tasks, leave_one_task_out, leave_one_task_partition_out, test_on_LOO_dataset)
#                 vali_frac - (if split_method is random/random_equal_tasks) Float: fraction of data to use for validation 
#                 test_frac - (if split_method is random/random_equal_tasks) Float: fraction of data to use for test
#                  LOO_task - (if split_method is leave_one_task_out) Int: task that makes up test set
#   LOO_num_task_partitions - (if split_method is leave_one_task_partition_out) Int: number of partitions to split task into
#        LOO_task_partition - (if split_method is leave_one_task_partition_out) Int: partition of tasks that makes up test set
#               LOO_dataset - (if split_method is test_on_LOO_dataset) Int: dataset that makes up test set
#          balance_datasets - (if split_method is test_on_LOO_dataset) Int: to balance the size of the training data from the test dataset against training data from other datasets
#                                 if balance_datasets <= 0, then no balancing is done
#                                 if balance_datasets = 1, then the training data from the test dataset will have same size as the training data from other datasets
#                                 if balance_datasets > 1, then the training data from the test dataset will be repeated balance_datasets times
#                ensemble_i - Int: 
#                 vali_seed - (if split_method is random/random_equal_tasks) Int: seed to use for random split
#                 test_seed - (if split_method is random/random_equal_tasks) Int: seed to use for random split
def format_data(datasets, 
                labels, label_ln_coefs={"chl":1, "tss":1, "cdom":0.1}, invalid_label_action="drop_if_any",
                norm_spectra=True,
                task_selection_criterion="site", min_site_size=5, task_selection_n_clusters=None,
                split_method="", vali_frac=0, test_frac=0, LOO_task=None, LOO_num_task_partitions=None, LOO_task_partition=None, LOO_dataset=None,
                balance_datasets=0, 
                ensemble_i=0, vali_seed=0, test_seed=0):
    
    if isinstance(datasets, pd.DataFrame):
        datasets = [datasets]
    if isinstance(labels, str):
        labels = [labels]

    # If using ensembles, change seed for each iteration
    if ensemble_i > 0:
        vali_seed += ensemble_i


    ######## Combine datasets ########

    
    for df in datasets:
        assert len(df) > 0 and len(df.columns) > 0, "ERROR: No data loaded in dataset {}.".format(df)

    for i in range(len(datasets)):
        datasets[i]["dataset"] = i
        datasets[i]["site"] = datasets[i]["site"].apply(lambda x: str(i) + "_" + str(x))

    data_df = pd.concat(datasets, ignore_index=True)

    colnames = set(datasets[0].columns)
    for df in datasets[1:]:
        colnames = colnames.intersection(df.columns)
    data_df = data_df[sorted(list(colnames))]


    ######## Format data ########


    ### Format features ###
    feature_colnames = [c for c in data_df.columns if c.startswith("feature_")]
    
    if norm_spectra:
        data_df[feature_colnames] = normalise_spectra(data_df[feature_colnames])
    data_df = data_df.dropna(how="any", subset=feature_colnames)
    #data_df[feature_colnames] = data_df[feature_colnames].dropna(how="any")
    data_df = data_df.drop([c for c in data_df.columns if c in feature_colnames and data_df[c].isna().any()], axis=1)

    ### Format labels ###
    label_colnames = [f"label_{label}" for label in labels]
    for label in labels:
        data_df[f"label_{label}"] = np.log(data_df[f"label_{label}"]+label_ln_coefs[label])
    
    if invalid_label_action == "drop_if_any":
        data_df = data_df.dropna(subset=label_colnames, how="any")
    elif invalid_label_action == "drop_if_all":
        data_df = data_df.dropna(subset=label_colnames, how="all")
    elif invalid_label_action == "impute":
        NotImplementedError("Imputation of labels is not implemented yet.")
    else:
        assert False, "ERROR: Invalid argument for invalid_label_action: {}.".format(invalid_label_action)
    assert len(labels) > 1 or invalid_label_action in ["drop_if_any", "drop_if_all"], "ERROR: invalid_label_action {} is not compatible with single labels.".format(invalid_label_action)


    ### Filter data ###
    data_df = data_df[~data_df.duplicated(subset=feature_colnames)]
    if min_site_size > 1:
        counts = data_df.groupby("site")["site"].transform('count') # ["label"], maybe label_colnames[0] if necessary
        data_df = data_df[counts >= min_site_size]
    

    ######## Task ########


    # From "Remote Sensing of Environment Remotely estimating total suspended solids concentration in clear to extremely turbid waters using a novel semi-analytical method"
    if task_selection_criterion == "TSS1":
        assert label=="tss", "ERROR: task_selection_method tss is not compatible with label {}".format(label)
        
        # OLCI
        def identify_task(row):
            if row["490"]>row["560"]:
                return 0
            elif row["490"]>row["620"]:
                return 1
            elif row["753.75"]>row["490"] and row["753.75"] > 0.01: 
                return 3
            else:
                return 2
        data_df["task"] = data_df.apply(identify_task, axis=1)
            
    # Naively cluster spectra
    elif task_selection_criterion == "cluster":
        assert not task_selection_n_clusters is None, "ERROR: task_selection_method cluster requires task_selection_n_clusters to be set."
        
        rrs_vals = data_df.drop(columns=label_colnames).values

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=task_selection_n_clusters)
        clusters = kmeans.fit_predict(rrs_vals)
        data_df["task"] = clusters
    
    # Tasks based on site data was collected
    elif task_selection_criterion == "site":
        #assert min_site_size >= 5, "ERROR: If grouping tasks on sites, you must have a minimum site size >= 5."
        
        # Replace sites of data from small sites with a custom site, "small site" which combines all small lakes
        #if min_site_size > 1 and min_site_action == "keep":
        #    data_df['site'] = data_df['site'].apply(lambda x: x if data_df['site'].value_counts()[x] >= min_site_size else f"{x[0]}_smallsite")

        # Sort the sites by (dataset, then) size, then assign tasks based on the order
        site_sizes = data_df.groupby(["dataset", "site"]).size().reset_index(name="size")
        sites_sorted = site_sizes.sort_values(by=["dataset", "size"], ascending=[True, False])
        site_mapping = {group: idx for idx, group in enumerate(sites_sorted["site"])}
    
        data_df["task"] = data_df["site"].map(site_mapping)
        data_df = data_df.drop("site", axis=1)

    else:
        raise NotImplementedError("Unknown task selection criterion {}.".format(task_selection_criterion))
        
    # Format data into X, y, t
    X = data_df[feature_colnames]
    y = data_df[label_colnames]
    T = data_df[["task"]]
    D = data_df[["dataset"]]
    T_names = [t for t in site_mapping]
    

    ######## Test/train/validation split ########


    # Random test/train/validation split, if random_equal_tasks then stratify on each task
    if split_method in ["random", "random_equal_tasks"]:
        assert test_frac > 0, "ERROR: No test fraction set with random split."
        
        num_data = len(y)
        num_vali = int(vali_frac * num_data)
        num_test = int(test_frac * num_data)
        
        indices = np.arange(num_data)

        stratify = T.values if split_method in ["random_equal_tasks"] else None
        indices_nottest, indices_test = train_test_split(indices, test_size=num_test, random_state=test_seed, stratify=stratify)
        if num_vali > 0:
            stratify = T.values[indices_nottest] if split_method == "random_equal_tasks" else None
            indices_train, indices_vali = train_test_split(indices_nottest, test_size=num_vali, random_state=vali_seed, stratify=stratify)    
        else:
            indices_train = indices_nottest
            indices_vali = np.array([])

    # Equivalent to random_equal_tasks for data from LOO_dataset, all other data goes into training set only
    elif split_method in ["test_on_LOO_dataset"]:
        assert not (LOO_dataset is None), "ERROR: No LOO_dataset set with test_on_LOO_dataset split."

        indices_testdataset = np.where(D.values == LOO_dataset)[0]
        indices_nottestdataset = np.where(D.values != LOO_dataset)[0]
        T_testdataset = T.values[indices_testdataset]
        T_nottestdataset = T.values[indices_nottestdataset]

        # Split indices from the test dataset into test/validation and test sets
        num_test = int(test_frac * len(T_testdataset))
        T_map = regroup_min_size_map(T_testdataset, 2) 
        stratify = [T_map[t] for t in T_testdataset.flatten()] # Group tasks with only one data point into a single group
        indices_nottest, indices_test, T_nottest, T_test = train_test_split(indices_testdataset, T_testdataset, test_size=num_test, random_state=test_seed, stratify=stratify)

        # Add indices from the not test dataset[s] into the training/validation set
        indices_nottest = np.concatenate((indices_nottest, indices_nottestdataset))
        T_nottest = np.concatenate((T_nottest, T_nottestdataset))

        # Separate training/validation sets
        num_vali = int(len(T_nottest) * vali_frac/(1-test_frac))
        if num_vali > 0:
            min_group_size = int(np.ceil((1-test_frac)/vali_frac))
            T_map = regroup_min_size_map(T_nottest, min_group_size)
            stratify = [T_map[t] for t in T_nottest.flatten()] # Group tasks with fewer than min_group_size data points into a single group
            indices_train, indices_vali = train_test_split(indices_nottest, test_size=num_vali, random_state=vali_seed, stratify=stratify)    
        else:
            indices_train = indices_nottest
            indices_vali = np.array([])

        # Balance the size of the training data from the test dataset and from other datasets
        if balance_datasets > 0 and len(indices_nottestdataset) > 0:
            NotImplementedError("Balancing datasets is not implemented.")
        #    if balance_datasets > 1:
        #        num_repeats = balance_datasets
        #    else:
        #        num_repeats = int(len(indices_nottestdataset)/len(indices_nottest))
        #    indices_nottest = np.repeat(indices_nottest, num_repeats, axis=0)
        #    T_nottest = np.repeat(T_nottest, num_repeats, axis=0)
        #    print("WARNING: Repeating test dataset {} times to balance training data.".format(num_repeats))


    # Test set is one task
    elif split_method in ["leave_one_task_out", "leave_one_task_partition_out"]:
        assert not (split_method == "leave_one_task_out" and LOO_task is None), "ERROR: No LOO_task set with leave_one_task_out split."
        assert not (split_method == "leave_one_task_partition_out" and LOO_task_partition is None), "ERROR: No LOO_task_partition set with leave_one_task_partition_out split."

        if split_method == "leave_one_task_out":
            indices_test = np.where(T.values == LOO_task)[0]
            indices_nottest = np.where(T.values != LOO_task)[0]
        else:
            indices_test = np.where(np.mod(T.values, LOO_num_task_partitions) == LOO_task_partition)[0]
            indices_nottest = np.where(np.mod(T.values, LOO_num_task_partitions) != LOO_task_partition)[0]
        
        num_vali = int(len(indices_nottest) * vali_frac/(1-test_frac))
        indices_train, indices_vali = train_test_split(indices_nottest, test_size=num_vali, random_state=vali_seed)

    else:
        assert False, "ERROR: Unknown SPLIT method: {}.".format(split_method)

    if balance_datasets and not split_method == "test_on_LOO_dataset":
        NotImplementedError("Balancing datasets is only implemented for test_on_LOO_dataset split method.")
    

    ######## Return data ########
    
    
    output_data_dict = {}

    # Full datasets
    output_data_dict["X_df"] = X
    output_data_dict["y_df"] = y
    output_data_dict["T_df"] = T
    output_data_dict["D_df"] = D
    output_data_dict["T_names"] = T_names

    # Partition datasets
    output_data_dict["X_train_df"] = X.iloc[indices_train]
    output_data_dict["y_train_df"] = y.iloc[indices_train]
    output_data_dict["T_train_df"] = T.iloc[indices_train]
    output_data_dict["D_train_df"] = D.iloc[indices_train]
    output_data_dict["X_vali_df"] = X.iloc[indices_vali]
    output_data_dict["y_vali_df"] = y.iloc[indices_vali]
    output_data_dict["T_vali_df"] = T.iloc[indices_vali]
    output_data_dict["D_vali_df"] = D.iloc[indices_vali]
    output_data_dict["X_test_df"] = X.iloc[indices_test]
    output_data_dict["y_test_df"] = y.iloc[indices_test]
    output_data_dict["T_test_df"] = T.iloc[indices_test]
    output_data_dict["D_test_df"] = D.iloc[indices_test]

    # Return inds of partitions (these aren't the same as indices of partition dfs, which are linked to original, unfiltered data)
    output_data_dict["inds_train"] = indices_train
    output_data_dict["inds_vali"] = indices_vali
    output_data_dict["inds_test"] = indices_test

    #output_data_dict["T_df"]["a"] = 1 # Dummy column for plotting
    #display_full(output_data_dict["T_df"].groupby("task").count())
    
    return output_data_dict
    