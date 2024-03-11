import os
import sys
import warnings

import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# Utility functions
from src.utility.utility_get_data import *
from src.MTL.metrics import *
from src.utility.utility_misc import *
from src.MDN.benchmarks.tss.SOLID.model import model as SOLID
from src.MDN.benchmarks.tss.Novoa.model import model as Novoa
from src.MDN.benchmarks.chl.OC.model import model4 as OC4
from src.MDN import image_estimates, get_sensor_bands
from src.MDN.product_estimation import apply_model

import numpy as np 

# Generic class that can train any of the baseline MTL architectures or traditional ML models
class train_model():

    def __init__(self, params, data_params):

        # Copy all parameters
        self.metrics = {}
        for key in params:
            setattr(self, key, params[key])
            self.metrics[key] = params[key]
        self.data_params = data_params

        # Set seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Run models and gather predictions
        if self.data_params["split_method"] == "leave_one_task_out":
            self.data_params["LOO_split"] = 0 
            self._model_pipeline() # Run with task 0 to determine self.num_tasks
            for t in range(1, self.num_task_models):
                self.data_params["LOO_task"] = t
                if self.is_MTL:
                    assert False, "ERROR: Not implemented, LOO with MTL is currently broken since we need to do LOO on a task model rather than a task? Or is it idk"
                self._model_pipeline()
        elif self.data_params["split_method"] == "leave_one_task_partition_out":
            for p in range(self.data_params["LOO_num_task_partitions"]):
                self.data_params["LOO_task_partition"] = p
                self._model_pipeline()
        else:
            self._model_pipeline()

        # Evaluate performance and plot results
        self._evaluate_performance()
        if self.plot:
            self.plot_results()


    # Initialise, train and record predictions from the model
    def _model_pipeline(self):

        if self.num_ensembles > 1:
            for i in range(self.num_ensembles):
                self.data_params["ensemble_i"] = i
                self._set_params()
                self._prepare_data()
                self._record_predictions()
        else:
            self._set_params()
            self._prepare_data()
            self._record_predictions()


    # Initialises parameters
    def _set_params(self):

        assert self.algo_name in ["OC4", "SOLID", "Novoa", "FICEK", "MDN_brandon"], "ERROR: Invalid model name: {}.".format(self.algo_name)
      
        # Default params
        p_default = {
                "is_classification": False,
                "loss_func_name": "mse",
                "normalise_task_training": False
            }

        # Record parameters
        for key in p_default:
            if not hasattr(self, key):
                setattr(self, key, p_default[key])
                self.metrics[key] = p_default[key]

    # Prepare the data for training
    def _prepare_data(self):

        data = get_data(**self.data_params, vali_seed=self.seed, test_seed=self.seed)

        for key in data:
            setattr(self, key, data[key])
            
        self.num_inputs = self.X_train_df.shape[1]
        self.num_tasks = self.T_df.values.max()+1
        self.num_data = len(self.T_df.index)
        self.num_train = len(self.T_train_df)

        self.feature_names = self.X_df.columns
        self.label_names = self.data_params["labels"]
        self.num_labels = len(self.label_names)

        # Convert to numpy arrays
        self.X_train = self.X_train_df.values
        self.X_vali = self.X_vali_df.values
        self.X_test = self.X_test_df.values
        self.y_train = self.y_train_df.values
        self.y_vali = self.y_vali_df.values
        self.y_test = self.y_test_df.values
        self.T_train = self.T_train_df.values
        self.T_vali = self.T_vali_df.values
        self.T_test = self.T_test_df.values

        assert self.X_vali.shape[0] == 0, "ERROR: Using validation set with pretrained model."

        # Task things
        y_train_vali = np.concatenate([self.y_train, self.y_vali])
        T_train_vali = np.concatenate([self.T_train, self.T_vali]).flatten()
        y = self.y_df.values
        T = self.T_df.values.flatten()
        self.T_unique = np.arange(self.num_tasks)
        self.T_count = [np.sum(T_train_vali == t) for t in self.T_unique]
        self.T_means = [np.mean(y_train_vali[T_train_vali == t]) if self.T_count[t] > 0 else np.mean(y_train_vali) for t in self.T_unique]
        self.y_mean = np.mean(y_train_vali)
        self.metrics["task_sizes"] = self.T_count
        self.metrics["task_means"] = self.T_means
        self.metrics["task_names"] = self.T_names


    # Get the predictions of the model on each partition
    def _record_predictions(self):


        if self.algo_name == "MDN_brandon":
            
            sensor = "OLCI"

            ws = [412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25, 708.75, 753.75, 778.75]

            X_train = self.X_train_df[["feature_{}".format(w) for w in ws]].values.reshape(-1, len(ws))
            X_test = self.X_test_df[["feature_{}".format(w) for w in ws]].values.reshape(-1, len(ws))

            #assert len(self.label_names) == 1, "ERROR: MDN_brandon only works with one label."
            product = 'chl,tss,cdom'#self.label_names#[0]#chl'#, 'tss', 'cdom', 'ad', 'ag', 'aph' #
            pred_train = apply_model(x_test=X_train, product=product, sensor=sensor, use_cmdline=False, no_load=False)[0]
            pred_test = apply_model(x_test=X_test, product=product, sensor=sensor, use_cmdline=False, no_load=False)[0]

            preds_train = [] 
            preds_test = []
            for i, product in enumerate(["chl", "tss", "cdom"]):
                if product in self.label_names:
                    preds_train.append(np.log(pred_train[:,i] + self.data_params["label_ln_coefs"][product]))
                    preds_test.append(np.log(pred_test[:,i] + self.data_params["label_ln_coefs"][product]))
            pred_train = np.stack(preds_train, axis=1)
            pred_test = np.stack(preds_test, axis=1)

        elif self.algo_name == "OC4":

            #Usage:
            #    chl = get_oc([Rrs443,Rrs490,Rrs510,Rrs555], algorithm)
            #where Rrs is an array of Rrs at respective wavelengths 
            #and algorithm is 'OC4' or 'KD2S' or ... (see below)

            sensor = "OLCI"
            ws = [442.5, 490, 510, 560]

            X_train_Rrs = self.X_train_df[["feature_{}".format(w) for w in ws]].values
            X_test_Rrs = self.X_test_df[["feature_{}".format(w) for w in ws]].values

            pred_train = OC4(X_train_Rrs, ws, sensor)
            pred_test = OC4(X_test_Rrs, ws, sensor)

            pred_train = np.log(pred_train + self.data_params["label_ln_coefs"]["chl"])
            pred_test = np.log(pred_test + self.data_params["label_ln_coefs"]["chl"])
            
        elif self.algo_name == "SOLID":
            
            assert self.label_names == ["tss"], "ERROR: SOLID only works with TSS."

            ws = [412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25]
            ws_algo = [411, 442, 490, 510, 560, 619, 664, 673, 681]
            print("TODO: fix this, double band")
            sensor = "OLCI"

            X_train = self.X_train_df[["feature_{}".format(w) for w in ws]].values
            X_test = self.X_test_df[["feature_{}".format(w) for w in ws]].values

            pred_train = SOLID(X_train, ws, sensor).reshape(-1, 1)
            pred_test = SOLID(X_test, ws, sensor).reshape(-1, 1)

            pred_train[pred_train < 0] = 0
            pred_test[pred_test < 0] = 0
            pred_train[pred_train > 1e6] = 1e6
            pred_test[pred_test > 1e6] = 1e6
            
            pred_train = np.log10(pred_train + self.data_params["label_ln_coefs"]["tss"])
            pred_test = np.log10(pred_test + self.data_params["label_ln_coefs"]["tss"])

            pred_train[np.isinf(pred_train)] = np.nan
            pred_test[np.isinf(pred_test)] = np.nan

            plt.scatter(self.y_train, pred_train, alpha=0.05)
            plt.plot([0, 10], [0, 10], ":")
            plt.show()

        elif self.algo_name == "Novoa":
            
            assert self.label_names == ["tss"], "ERROR: Novoa only works with TSS."

            ws = [490, 560, 665]
            ws_algo = [482, 561, 665]
            print("TODO: fix this, double band")
            sensor = "OLCI"

            X_train = self.X_train_df[["feature_{}".format(w) for w in ws]].values
            X_test = self.X_test_df[["feature_{}".format(w) for w in ws]].values

            pred_train = Novoa(X_train, ws, sensor).reshape(-1, 1)
            pred_test = Novoa(X_test, ws, sensor).reshape(-1, 1)

            pred_train = np.log(pred_train + self.data_params["label_ln_coefs"]["tss"])
            pred_test = np.log(pred_test + self.data_params["label_ln_coefs"]["tss"])

            plt.scatter(self.y_train, pred_train, alpha=0.05)
            plt.plot([0, 10], [0, 10], ":")
            plt.show()

        elif self.algo_name == "FICEK":

            assert self.label_names == ["cdom"], "ERROR: FICEK only works with cdom."

            ws = [560, 665]
            ws_algo = [570, 655] # Should be these, but OLCI doesn't have them

            X_train = self.X_train_df[["feature_{}".format(w) for w in ws]].values
            X_test = self.X_test_df[["feature_{}".format(w) for w in ws]].values

            def FICEK_OLCI(Rrs):
                ratio = Rrs[:,0] / np.maximum(Rrs[:,1], 1e-3)
                return 3.65 * ratio ** -1.93

            pred_train = FICEK_OLCI(X_train).reshape(-1, 1)
            pred_test = FICEK_OLCI(X_test).reshape(-1, 1)

            pred_train = np.log(pred_train + self.data_params["label_ln_coefs"]["cdom"])
            pred_test = np.log(pred_test + self.data_params["label_ln_coefs"]["cdom"])

        else:

            NotImplementedError("Model {} not implemented yet".format(self.algo_name))

        ### Record predictions ###

        # Check prediction storage lists exist
        if not hasattr(self, "pred_train"):        
            self.pred_train = [[] for i in range(self.num_data)]
            self.pred_vali = [[] for i in range(self.num_data)]
            self.pred_test = [[] for i in range(self.num_data)]


        # If not using ensemble, record results directly
        if self.num_ensembles <= 1:
            for i in range(len(self.inds_train)):
                self.pred_train[self.inds_train[i]].append(pred_train[i])
            for i in range(len(self.inds_vali)):
                self.pred_vali[self.inds_vali[i]].append(pred_vali[i])
            for i in range(len(self.inds_test)):
                self.pred_test[self.inds_test[i]].append(pred_test[i])


        # If using ensemble, record results in temporary lists after each ensemble model is run
        else:
            if not hasattr(self, "ensemble_pred_train"):
                self.ensemble_pred_train = [[] for i in range(self.num_data)]
                self.ensemble_pred_vali = [[] for i in range(self.num_data)]
                self.ensemble_pred_test = [[] for i in range(self.num_data)]
            for i in range(len(self.inds_train)):
                self.ensemble_pred_train[self.inds_train[i]].append(pred_train[i])
            for i in range(len(self.inds_vali)):
                self.ensemble_pred_vali[self.inds_vali[i]].append(pred_vali[i])
            for i in range(len(self.inds_test)):
                self.ensemble_pred_test[self.inds_test[i]].append(pred_test[i])
            
            # After last ensemble, add results to main lists
            if self.data_params["ensemble_i"] == self.num_ensembles-1:
                # Add average of predictions for each data point to results
                if self.agg_ensembles:
                    for i in range(self.num_data):
                        if len(self.ensemble_pred_train[i]) > 0:
                            self.pred_train[i].append(np.mean(np.stack(self.ensemble_pred_train[i]), axis=0))
                        if len(self.ensemble_pred_vali[i]) > 0:
                            self.pred_vali[i].append(np.mean(np.stack(self.ensemble_pred_vali[i]), axis=0))
                        if len(self.ensemble_pred_test[i]) > 0:
                            self.pred_test[i].append(np.mean(np.stack(self.ensemble_pred_test[i]), axis=0))
                # Add multiple predictions for each data point to results
                else:
                    self.pred_train = self.ensemble_pred_train
                    self.pred_vali = self.ensemble_pred_vali
                    self.pred_test = self.ensemble_pred_test

    # Evaluate the performance of the predictions generated by the model. End of model pipeline.
    def _evaluate_performance(self):

        for partition in ["train", "vali", "test"]:

            if partition == "train":
                pred = self.pred_train.copy()
            elif partition == "vali":
                pred = self.pred_vali.copy()
            elif partition == "test":
                pred = self.pred_test.copy()

            ### Format preds ###
            # Each element doesn't necessarily have the same number of predictions, for example if using leave-one-out
            # test set, each test point will have one prediction but train/vali will likely have multiple.
            # Therefore, we need to pad the shorter ones with nans, so that we can convert to a numpy array

            # Convert list of lists into a list of 2D numpy arrays of shape [num_preds, num_labels]
            for i in range(len(pred)):
                if len(pred[i]) > 0:
                    pred[i] = np.stack(pred[i], axis=0)
                else:
                    pred[i] = np.zeros((0, self.num_labels))

            # Convert list of 2D arrays into one 3D array of shape [num_data, max_preds, num_labels]
            max_preds = max(p.shape[0] for p in pred)
            y_pred = np.stack([np.vstack((p, np.full((max_preds - p.shape[0], self.num_labels), np.nan))) for p in pred])

            # Format true values for comparison against multiple predictions per data point (same shape as pred)
            y_true = self.y_df.values.copy().reshape(-1, 1, self.num_labels)
            y_true = np.repeat(y_true, max_preds, axis=1)

            # Find means of each task, and format them for comparison against multiple predictions per data point (same shape as pred)
            T = self.T_df.values.copy().flatten()
            y_mean = np.array([self.T_means[t] for t in T]).reshape(-1, 1, self.num_labels)
            y_mean = np.repeat(y_mean, max_preds, axis=1)

            # Format tasks for comparison against multiple predictions per data point [shape (num_data, max_preds)]
            T = np.repeat(T.reshape(-1, 1), max_preds, axis=1)

            metric_funcs = {"rmse": root_mean_squared_error, "rmsle": root_mean_squared_log_error, "mae": mean_absolute_error, "mape": median_absolute_percentage_error, 
                            "slope": slope, "bias": bias, "mdsa": median_symmetric_accuracy, "sspb": symmetric_signed_percentage_bias, 
                            "r2": r2, "r2_intra_group": r2_intra_group}#, "r2_group_mean": r2_group_mean}
                
            for i in range(len(self.label_names)):
                label_name = self.label_names[i]
                y_true_i = y_true[:,:,i]
                y_pred_i = y_pred[:,:,i]
                y_mean_i = y_mean[:,:,i]

                # Un-log the data (base e)
                y_true_i = np.exp(y_true_i)
                y_pred_i = np.exp(y_pred_i)
                y_mean_i = np.exp(y_mean_i) 
                
                ### Record results ###
                setattr(self, f"pred_{label_name}_{partition}", y_pred_i)
                setattr(self, f"true_{label_name}_{partition}", y_true_i)
                setattr(self, f"mean_{label_name}_{partition}", y_mean_i)

                ### Compute metrics ###  
                for metric in metric_funcs:
                    
                    if np.any(~np.isnan(y_pred_i)):
                        if metric in ["r2", "r2_intra_group"]:
                            self.metrics[f"{metric}_{label_name}_{partition}"] = metric_funcs[metric](y_true_i, y_pred_i, y_mean_i)
                        else:
                            self.metrics[f"{metric}_{label_name}_{partition}"] = metric_funcs[metric](y_true_i, y_pred_i)
                    
                    for t in range(self.num_tasks):
                        y_true_i_t = y_true_i[T == t]
                        y_pred_i_t = y_pred_i[T == t]
                        y_mean_i_t = y_mean_i[T == t]
                            
                        if np.any(~np.isnan(y_pred_i_t)):
                            if metric in ["r2", "r2_intra_group"]:
                                self.metrics[f"{metric}_{label_name}_{partition}_task{t:03}"] = metric_funcs[metric](y_true_i_t, y_pred_i_t, y_mean_i_t)
                            elif metric not in ["r2_intra_group", "r2_group_mean"]:
                                self.metrics[f"{metric}_{label_name}_{partition}_task{t:03}"] = metric_funcs[metric](y_true_i_t, y_pred_i_t)


    
    def plot_results(self):


        ### Predictions vs. real plot ###
        fig, axs = plt.subplots(self.num_labels, 3, figsize=(16, 5*self.num_labels))
        cMap = plt.get_cmap('hsv')
        cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(self.T_unique))
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cMap)

        T = self.T_df.values.flatten()

        for label_i in range(self.num_labels):

            label_name = self.label_names[label_i]

            for partition_i, partition_name in enumerate(["train", "vali", "test"]):

                y_true = getattr(self, f"true_{label_name}_{partition_name}")
                y_pred = getattr(self, f"pred_{label_name}_{partition_name}")

                if y_true.size < 1:
                    continue

                #print(f"{label_name} {partition_name} RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.4f}")
                #print("y_true: ", y_true)
                #print("y_pred: ", y_pred)

                min_val = np.nanmin(np.concatenate([y_true, y_pred]))
                max_val = np.nanmax(np.concatenate([y_true, y_pred]))
                axs.flat[label_i*3+partition_i].plot([min_val, max_val], [min_val, max_val], ":", alpha=0.5)
                axs.flat[label_i*3+partition_i].set(xlabel="Real "+label_name, ylabel="Predicted "+label_name, title=partition_name+" dataset",
                                                    xscale="log", yscale="log")
                
                for t in range(self.num_tasks):

                    y_true_task = y_true[T == t].flatten()
                    y_pred_task = y_pred[T == t].flatten()

                    col = scalarMap.to_rgba(t)
                    axs.flat[label_i*3+partition_i].scatter(y_true_task, y_pred_task, label=f"Task {t}", s=15, alpha=0.5, color=col)

        plt.show()


