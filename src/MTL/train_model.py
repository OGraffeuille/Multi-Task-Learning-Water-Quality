import os
import sys
import warnings

import math
import random
import numpy as np
import glob
import datetime
import time
import datetime
import copy
import json

import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import xgboost as xgb

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim

# MTL algorithms
from . import MTL

# Utility functions
from ..utility.utility_get_data import *
from .metrics import *
from ..utility.utility_misc import *
import numpy as np 

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        torch.manual_seed(self.seed)

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
                self._initialise_model()
                self._train_model()
                self._record_predictions()
        else:
            self._set_params()
            self._prepare_data()
            self._initialise_model()
            self._train_model()
            self._record_predictions()


    # Initialises parameters
    def _set_params(self):

        assert self.algo_name in ["MLP_STL", "MDN_STL", "MLP_MTL", "MDN_MTL", "linear_regression", "RF", "SVM", "XGBoost", "OC4", "Naive"], "ERROR: Invalid model name: {}.".format(self.algo_name)
      
        self.is_network_model = any(self.algo_name.startswith(s) for s in ["MLP", "MDN"])

        # Default params
        p_default = {
                "is_classification": False,
                "loss_func_name": "mse",
                "normalise_task_training": False
            }
        if self.is_network_model:
            p_default.update({
                "vali_epoch_freq": 5,
                "vali_epoch_delay": 20,
                "max_epochs": 2000,
                "batch_size": 64,
                "batch_shuffle": True, 
                "testdataset_weight": 1   
            })
        if "RF" in self.algo_name:
            p_default.update({
                "rf_n_trees": 100,
                "rf_criterion": "mse",
                "rf_max_depth": None,
                "rf_min_samples_split": 2,
                "rf_min_samples_leaf": 1,
                "rf_max_features": 1
            })
        elif "XGBoost" in self.algo_name:
            p_default.update({
                'xgb_objective': 'reg:squarederror', 
                'xgb_lr': 0.3,
                'xgb_max_depth': 6
            })
        elif "SVM" in self.algo_name:
            p_default.update({
                "svm_kernel": "rbf",
                "svm_gamma": "scale",
                "svm_C": 1.0
            })

        # Record parameters
        for key in p_default:
            if not hasattr(self, key):
                setattr(self, key, p_default[key])
                self.metrics[key] = p_default[key]

        # Special parameters
        if self.is_network_model:
            self.shared_layer_sizes = self.arch[0]
            self.task_layer_sizes = self.arch[1]
            self.metrics["shared_layers"] = ",".join([str(size) for size in self.shared_layer_sizes])
            self.metrics["task_layers"] = ",".join([str(size) for size in self.task_layer_sizes])

        self.is_MTL = self.is_network_model and len(self.task_layer_sizes) > 0
        self.is_MDN = self.is_network_model and "MDN" in self.algo_name


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
        self.D_train = self.D_train_df.values

        print("Sizes: Train: {}, Vali: {}, Test: {}".format(self.X_train.shape[0], self.X_vali.shape[0], self.X_test.shape[0]))

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

        # Get task models from tasks (if MTL): Tasks with fewer than min_site_size instances are put to task 0
        if self.is_MTL:
            self.T_models_map = regroup_min_size_map(T_train_vali, self.min_site_size, self.T_unique)
            self.num_task_models = len(self.T_models_map)
        else:
            self.num_task_models = 1
                    
        # Convert to tensors & normalise
        if self.is_network_model:

            self.X_train = torch.from_numpy(self.X_train).float().to(device)
            self.X_vali = torch.from_numpy(self.X_vali).float().to(device)
            self.X_test = torch.from_numpy(self.X_test).float().to(device)
            self.y_train = torch.from_numpy(self.y_train).float().to(device)
            self.y_vali = torch.from_numpy(self.y_vali).float().to(device)
            self.y_test = torch.from_numpy(self.y_test).float().to(device)
            self.T_train = torch.from_numpy(self.T_train).long().to(device)
            self.T_vali = torch.from_numpy(self.T_vali).long().to(device)
            self.T_test = torch.from_numpy(self.T_test).long().to(device)
            self.D_train = torch.from_numpy(self.D_train).long().to(device)

            # X
            self.X_mu = torch.mean(torch.cat([self.X_train, self.X_vali], axis=0), axis=0)
            self.X_std = torch.std(torch.cat([self.X_train, self.X_vali], axis=0), axis=0)
            self.X_std[self.X_std == 0] = 1
            self.X_train = torch.div(torch.add(self.X_train, -self.X_mu), self.X_std)
            self.X_vali = torch.div(torch.add(self.X_vali, -self.X_mu), self.X_std)
            self.X_test = torch.div(torch.add(self.X_test, -self.X_mu), self.X_std)

            # y: normalise regression target
            self.y_vali = torch.from_numpy(self.y_vali_df.values).float().to(device)#.view(-1,1)
            y_nottest = torch.cat([self.y_train, self.y_vali], axis=0)
            self.y_mu = torch.mean(y_nottest[~torch.isnan(y_nottest)], axis=0)
            self.y_std = torch.std(y_nottest[~torch.isnan(y_nottest)], axis=0)
            self.y_train = torch.div(torch.add(self.y_train, -self.y_mu), self.y_std)
            self.y_vali = torch.div(torch.add(self.y_vali, -self.y_mu), self.y_std)
            self.y_test = torch.div(torch.add(self.y_test, -self.y_mu), self.y_std)


            self.dataset_train = torch.utils.data.TensorDataset(self.X_train, self.y_train, self.T_train, self.D_train)
            self.dataset_vali = torch.utils.data.TensorDataset(self.X_vali, self.y_vali, self.T_vali)
            self.dataset_test = torch.utils.data.TensorDataset(self.X_test, self.y_test, self.T_test)

            self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.batch_shuffle)
        
        elif self.algo_name == "XGBoost":
            
            self.D_train = xgb.DMatrix(data=self.X_train, label=self.y_train)
            self.X_vali = xgb.DMatrix(data=self.X_vali)
            self.X_test = xgb.DMatrix(data=self.X_test)
            

    # Initialise the model architecture and other relevant variables    
    def _initialise_model(self):

        if self.is_network_model:
        
            if self.is_MDN:
                self.model = MTL.MDN_Net(self.num_inputs, self.shared_layer_sizes, self.task_layer_sizes, self.num_task_models, self.num_gaussians).to(device)
                self.y_min = torch.min(self.y_train)
                self.y_max = torch.max(self.y_train)
            else:
                self.model = MTL.MTL_Net(self.num_inputs, self.shared_layer_sizes, self.task_layer_sizes, self.num_task_models).to(device)

            parameter_dict = [{"params": self.model.parameters()}]
            self.optimizer = optim.Adam(parameter_dict, lr=self.lr)

            if self.is_MDN:
                self.loss_func = self.gaussian_mixture_NLL
                self.eval_metric = mse_loss
            else:
                self.loss_func = mse_loss
                self.eval_metric = mse_loss

        else:

            if "linear_regression" in self.algo_name:
                self.model = LinearRegression()
            elif "RF" in self.algo_name:
                self.model = RandomForestRegressor(n_estimators=self.rf_n_trees, criterion=self.rf_criterion, max_depth=self.rf_max_depth, 
                                              min_samples_split=self.rf_min_samples_split, min_samples_leaf=self.rf_min_samples_leaf,
                                              max_features=self.rf_max_features)#, random_state=self.seed)
            elif "SVM" in self.algo_name:
                self.model = SVR(kernel=self.svm_kernel, gamma=self.svm_gamma, C=self.svm_C)#, random_state=self.seed)
            elif "XGBoost" in self.algo_name or "Naive" in self.algo_name:
                pass # Don't need to define model here
            else:
                raise ValueError('Invalid model_type: {}.'.format(self.algo_name))

          
    # Optimise a ML model
    # Single task models are also optimised here, using same task ID for all datapoints
    def _train_model(self):

        # If deep learning, train over epochs
        if self.is_network_model:
            
            print(f"MTL - Learning rate: {self.lr:.6f}, seed: {self.seed}, {self.num_inputs} features")
            if self.data_params["split_method"] == "leave_one_task_out":
                print("Leaving out tasks {}/{}.".format(self.data_params["LOO_task"]+1, self.num_task_models))
            elif self.data_params["split_method"] == "leave_one_task_partition_out":
                print("Leaving out task partition {}/{}.".format(self.data_params["LOO_task_partition"]+1, self.data_params["LOO_num_task_partitions"]))

            self.loss_train_hist = []
            self.loss_vali_hist = []

            epoch = 0
            while epoch < self.max_epochs:

                # Training data
                train_loss_epoch = 0
                for data in self.train_loader:
                    X_local, y_local, T_local, D_local = data

                    self.optimizer.zero_grad()
                    loss = self._run_model(X_local, y_local, T_local, D_local, train_mode=True) * len(y_local)
                    
                    loss.backward()

                    if self.is_MDN and check_for_nan_grad(self.model):
                        print("NaN gradient detected in MDN model - this occurs due to rare numerical instability, skipping batch.")
                        continue
                    
                    self.optimizer.step()
                    
                    train_loss_epoch += loss.detach().cpu().numpy()

                loss_train = train_loss_epoch / self.num_train
                self.loss_train_hist.append(loss_train)
                self.metrics["loss_train"] = loss_train

                # Validation
                if np.mod(epoch, self.vali_epoch_freq) == 0:

                    loss_vali = self._run_model(self.X_vali, self.y_vali, self.T_vali)
                    self.loss_vali_hist.append(loss_vali)
                    self.metrics["loss_vali"] = loss_vali.item()

                    if self.verbose:
                        print("LR: {:.6f}, Epoch: {:4d}/{:4d}, Train | Validation Loss: {:.4f} | {:.4f}".format(self.lr, epoch, self.max_epochs, loss_train, loss_vali))

                    delay = math.ceil(self.vali_epoch_delay / self.vali_epoch_freq)
                    if len(self.loss_vali_hist) > delay and self.loss_vali_hist[-1] >= self.loss_vali_hist[-1-delay]:
                        break

                epoch += 1

            self.metrics["loss_test"] = self._run_model(self.X_test, self.y_test, self.T_test).item()

            self.metrics["num_epochs"] = epoch

        # If not deep learning, train directly
        else:

            if "linear_regression" in self.algo_name:
                self.model.fit(self.X_train, self.y_train)
            elif "RF" in self.algo_name:
                self.model.fit(self.X_train, self.y_train)
            elif "SVM" in self.algo_name:
                self.model.fit(self.X_train, self.y_train)
            elif "XGBoost" in self.algo_name:
                params = {'objective': self.xgb_objective, 'learning_rate': self.xgb_lr, 'max_depth': self.xgb_max_depth}
                self.model = xgb.train(params=params, dtrain=self.D_train)
            elif "Naive":   
                pass # Don't need to train, just return task means
            else:
                raise ValueError('Invalid model_type: {}.'.format(self.algo_name))
            

    # Computes pred = f(X), compares it to y to give loss (and backpropagate if necessary)
    def _run_model(self, X, y, T, D=None, train_mode=False, eval_mode=False):

        # Change network mode
        if not train_mode:
            torch.set_grad_enabled(False)
            self.model.eval()

        # Compute Prediction for all tasks
        pred = self.model(X)
        
        # For compute efficiency, the MTL model makes one prediction per task for all data points
        # so, we need to extract the appropriate task prediction for each data point
        if self.is_MTL:
            # Select appropriate task model for each task (T_models_map is necessary to group up small tasks into one model)
            T = np.array([self.T_models_map[t.item()] for t in T])
            
            # Only keep relevant task predictions
            pred = pred.view(len(T), self.num_task_models, -1)
            pred = pred[torch.arange(len(T)), T.flatten(), :]

        # Get point prediction from mixture terms if MDN
        if eval_mode and self.is_MDN:
            pred = self._gaussian_mixture_point(pred)

        # Un-normalise regression target if required
        if eval_mode and not self.is_classification:
            pred = pred * self.y_std + self.y_mu
            y = y * self.y_std + self.y_mu

        # Compute loss
        if eval_mode:
            loss = self.eval_metric(y, pred)
        else:
            loss = self.loss_func(y, pred)
        
        # Weight loss from test datasets more heavily if required
        if self.testdataset_weight > 1 and train_mode:
            loss_weight = torch.ones_like(loss)
            loss_weight[D.flatten() == 0] = self.testdataset_weight
            loss_weight = loss_weight / torch.mean(loss_weight)
            loss = loss * loss_weight
                        
        # Multiply losses such that tasks with fewer instances get greater loss
        if self.normalise_task_training:
            mult = 1 / self.T_count[T]
            loss = loss * mult / torch.mean(mult)

        loss = torch.mean(loss)
        
        # Change network mode if necessary
        if not train_mode:
            torch.set_grad_enabled(True)
            self.model.train()

        if eval_mode:
            return loss, pred.detach().cpu().numpy()
        else:
            return loss


    # Get the predictions of the model on each partition
    def _record_predictions(self):

        if self.is_network_model:

            _, pred_train = self._run_model(self.X_train.float(), self.y_train, self.T_train, eval_mode=True)
            _, pred_vali = self._run_model(self.X_vali.float(), self.y_vali, self.T_vali, eval_mode=True)

            if self.is_MTL and self.data_params["split_method"] in ["leave_one_task_out", "leave_one_task_partition_out"]:
                T_imputed = self._impute_tasks(self.X_test, self.T_test)#, self.data_params["LOO_task"])
                acc_test, pred_test = self._run_model(self.X_test.float(), self.y_test, T_imputed, eval_mode=True)
            else:
                acc_test, pred_test = self._run_model(self.X_test.float(), self.y_test, self.T_test, eval_mode=True)
            print("{} features, LR: {}, seed: {}, epochs: {}, acc: {:.6f}".format(self.num_inputs, self.lr, self.seed, self.metrics["num_epochs"], acc_test))
            
        elif self.algo_name == "Naive":

            pred_train = np.array([self.T_means[t] for t in self.T_train.flatten()]).reshape(-1,1)
            pred_test = np.array([self.T_means[t] for t in self.T_test.flatten()]).reshape(-1,1)
            
        elif self.algo_name == "XGBoost":

            pred_train = self.model.predict(self.D_train).reshape(-1,1)
            pred_vali = self.model.predict(self.X_vali).reshape(-1,1)
            pred_test = self.model.predict(self.X_test).reshape(-1,1)

            #print(pred_train.shape, pred_vali.shape, pred_test.shape)

        else:

            pred_train = self.model.predict(self.X_train).reshape(-1,1)
            pred_vali = self.model.predict(self.X_vali).reshape(-1,1)
            pred_test = self.model.predict(self.X_test).reshape(-1,1)    

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
                            self.pred_train[i].append(np.median(np.stack(self.ensemble_pred_train[i]), axis=0))
                        if len(self.ensemble_pred_vali[i]) > 0:
                            self.pred_vali[i].append(np.median(np.stack(self.ensemble_pred_vali[i]), axis=0))
                        if len(self.ensemble_pred_test[i]) > 0:
                            self.pred_test[i].append(np.median(np.stack(self.ensemble_pred_test[i]), axis=0))
                # Add multiple predictions for each data point to results
                else:
                    self.pred_train = self.ensemble_pred_train
                    self.pred_vali = self.ensemble_pred_vali
                    self.pred_test = self.ensemble_pred_test


    # Evaluate the performance of the predictions generated by the model. End of model pipeline.
    # Predictions at this point are in log space. We exponentiate them to get actual predictions. Note that most evaluation metrics will log10 the predictions.
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
            # Note that task means are based on the training and validation set (ex. test set) for fair evaluation of R2 metrics
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
                #y_grand_mean = np.exp(self.y_mean)
                
                ### Record results ###
                setattr(self, f"pred_{label_name}_{partition}", y_pred_i)
                setattr(self, f"true_{label_name}_{partition}", y_true_i)
                setattr(self, f"mean_{label_name}_{partition}", y_mean_i)

                ### Compute metrics ###  
                for metric in metric_funcs:
                    
                    if np.any(~np.isnan(y_pred_i)):
                        #if metric in ["r2_group_mean"]: #unused
                        #    self.metrics[f"{metric}_{label_name}_{partition}"] = metric_funcs[metric](y_true_i, y_pred_i, y_mean_i, T)
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

    # Computes negative log loss of gaussian mixture  
    def gaussian_mixture_NLL(self, y, pred):
        if y.shape[0] < 1: # If empty predictions e.g. no validation set
            return y
        mu = pred[:, :self.num_gaussians]
        sigma = pred[:, self.num_gaussians:2*self.num_gaussians]
        pi = pred[:, 2*self.num_gaussians:]

        mu = mu.view(-1, self.num_gaussians, 1)
        sigma = sigma.view(-1, self.num_gaussians, 1)

        comp = D.Independent(D.Normal(mu, sigma), 1)
        mix = D.Categorical(pi)
        gmm = D.MixtureSameFamily(mix, comp)
        nll = -gmm.log_prob(y)
        
        # Replace likelihoods that were smaller than e^-10 with e^-10 to avoid nans
        nll[nll > 10] = 10
        
        return nll

    def _gaussian_mixture_point(self, pred, n_bins=1001):
        
        mu = pred[:, :self.num_gaussians]
        sigma = pred[:, self.num_gaussians:2*self.num_gaussians]
        pi = pred[:, 2*self.num_gaussians:]
        
        # Return mean of gaussian if mixture has just one component
        if mu.shape[1] == 1:
            y_opt = mu[:, 0]

        else:

            # Returns mean of mixture distribution
            weighted_means = pi * mu
            means = torch.sum(weighted_means, axis=1)#.detach().cpu().numpy()
            
            # Returns mean of most likely component of distribution (mu of distribution with greatest pi)
            inds = torch.argmax(pi, axis=1).view(-1,1)
            MLCs = torch.gather(mu, 1, inds)# .detach().cpu().numpy()

            # Returns mode of mixture distribution, estimated with bins
            gmm_bins = self._get_bins(n_bins)
            gmm_pdf = self._gaussian_mixture_pdf(mu, sigma, pi, gmm_bins)
            
            ind_modes = torch.argmax(gmm_pdf, axis=1)
            modes = gmm_bins[ind_modes]#.detach().cpu().numpy()
            
        # Convert before returning estimates
        return modes.view(-1, 1) 
    
    
    # Returns [num_gaussians x num_bins] array of 
    def _gaussian_mixture_pdf(self, mu, sigma, pi, bins):
        n_mixtures = mu.shape[0]
        n_gaussians = mu.shape[1]
        n_bins = len(bins)
        gmm_pdf = torch.zeros([n_mixtures, n_bins])
        
        for k in range(n_gaussians):
            gmm_pdf = gmm_pdf + pi[:, k].view(-1, 1) * self._gaussian_pdf(mu[:, k].view(-1, 1), sigma[:, k].view(-1, 1), bins)
        return gmm_pdf
    
    
    # Returns 2D Gaussian [num_distributions x num_bins]
    def _gaussian_pdf(self, mu, sigma, bins):
        sqrt_2_pi = 2.5066283
        val = 1 / (sigma * sqrt_2_pi) * torch.exp(-torch.pow(bins - mu, 2) / (2 * sigma * sigma))
        return val
    

    # Get bins to compute output distributions
    def _get_bins(self, n_bins=None, return_stepsize=False):
        if n_bins is None:
            n_bins = self.b
        min_val = self.y_min - 0.1*(self.y_max - self.y_min)
        max_val = self.y_max + 0.1*(self.y_max - self.y_min)
        if return_stepsize:
            return torch.linspace(min_val, max_val, n_bins), (max_val - min_val) / (n_bins - 1)
        else:
            return torch.linspace(min_val, max_val, n_bins)

    
    def plot_results(self):


        ### Learning curve plot ###
        if self.is_network_model:
            fig, axs = plt.subplots(1, 1, figsize=(16, 5))
            axs.plot(self.loss_train_hist, label="Train lr={:.5f}".format(self.lr))#, c="C"+str(seed_i), linestyle=":")
            axs.plot(np.arange(len(self.loss_vali_hist))*self.vali_epoch_freq, self.loss_vali_hist, label="Vali")#, c="C"+str(seed_i))

            axs.set(xlabel="Epochs", ylabel="Loss", title="Loss curve")
            axs.legend()
            plt.show()


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
                a_i = label_i*3+partition_i
                axs.flat[a_i].plot([min_val, max_val], [min_val, max_val], ":", alpha=0.5)
                axs.flat[a_i].set(xlabel="Real "+label_name, ylabel="Predicted "+label_name, title=partition_name+" dataset",
                                                    xscale="log", yscale="log")
                
                for t in range(self.num_tasks):

                    if t == 0:
                        print("WARNING: Only plotting a subset of tasks (for clarity).")

                    y_true_task = y_true[T == t].flatten()
                    y_pred_task = y_pred[T == t].flatten()

                    col = scalarMap.to_rgba(t)
                    axs.flat[a_i].scatter(y_true_task, y_pred_task, label=f"Task {t}", s=15, alpha=0.5, color=col)

        plt.show()



# Used for fixing occasional NaNs in MDN training
def check_for_nan_grad(model):
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False

def check_for_nan_params(model):
    for param in model.parameters():
        if torch.isnan(param).any():
            return True
    return False

def check_for_big_grad(model):
    for param in model.parameters():
        if param.grad is not None and torch.max(torch.abs(param.grad)) > 1e3:
            return True
    return False