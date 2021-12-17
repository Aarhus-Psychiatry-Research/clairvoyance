#!/usr/bin/env python
# coding: utf-8

# <img src="figure/clairvoyance_logo.png">
#
# # Clairvoyance: Time-series prediction
#
# ## ML-AIM (http://vanderschaar-lab.com/)
#
# This notebook describes the user-guide of a time-series predictions application using Clairvoyance framework. Time-series prediction is defined as following: utilize both static and temporal features to predict certain labels in the future. For instance, using the temporal data (vitals, lab tests) and static data (demographic information), we predict 'whether the patient will die at the end of hospital stay' or 'whether the patient will get ventilator after 4 hours'.
# - One-shot prediction: Predict the patient state at the end of the time-series at certain time point.
#   - Example: Predict patient mortality (at the end of the hospital stays) after 24 hours from the admission.
# - Rolling window (online) prediction:
#   - Example: Predict ventilator after 24 hours from the current time point.
#
# <img src="figure/time-series-prediction-definition.png">
#
# To run this tutorial, you need:
# ### Temporal and static datasets for training and testing
#
# If users come with their own temporal and static datasets for training and testing, the users should save those files as 'data_name_temporal_train_data_eav.csv.gz', 'data_name_static_train_data.csv.gz', 'data_name_temporal_test_data_eav.csv.gz', 'data_name_static_test_data.csv.gz' in '../datasets/data/data_name/' directory.
#
#
# ### Prerequisite
# Clone https://github.com/jsyoon0823/time-series-automl.git to the current directory.

# ## Time-series prediction pipeline summary
#
# <img src="figure/time-series-prediction-block-diagram.png">
#
# ### Step 1: Load dataset
#   - Extract csv files from the original raw datasets in ../datasets/data/data_name/ directory.
#
# ### Step 2: Preprocess dataset
#   - Preprocessing the raw data using various filters such as (1) replacing negative values to NaN, (2) do one-hot encidng for certain features, (3) do normalization.
#
# ### Step 3: Define problem
#   - Set the time-series prediction problem that we want to solve. Set the problem (whether it is one-shot or online prediction), set the label, set the maximum sequence length, and set the treatment features. We also define the metric for evaluation and the task itself (whether classification or regression).
#
# ### Step 4: Impute dataset
#   - Impute missing values in the preprocessed static and temporal datasets and return complete datasets.
#
# ### Step 5: Feature selection
#   - Select the relevant static and temporal features to the labels. You can skip the feature selection (set feature selection method = None).
#
# ### Step 6: Time-series model fit and predict
#   - After finishing the data preparation, we define the predictive models and train the model using the training dataset. After training, we use the trained model to predict the labels of the testing dataset.
#
# ### Step 7: Estimate uncertainty
#   - Estimate uncertainty of the predictor models and returns the uncertainty of the predictions.
#
# ### Step 8: Interpret predictions
#   - Interpret the trained predictor model and return the instance-wise feature and temporal importance.
#
# ### Step 9: Visualize results
#   - Visualize the various results such as performance, predictions, uncertainties, and interpretations.

#

# ## Step 0: Import necessary packages
#
# Import necessary packages for the entire tutorials.

# In[1]:


# Necessary packages
# from __future__ import absolute_importb
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append("../")


# In[ ]:


from utils import PipelineComposer


# ## Step 1: Load dataset
#
# Extract temporal and static datasets from 'data_name_temporal_train_data_eav.csv.gz', 'data_name_static_train_data.csv.gz', 'data_name_temporal_test_data_eav.csv.gz', 'data_name_static_test_data.csv.gz' in '../datasets/data/data_name/' directory.
#
# - CSVLoader: Load csv files from the original raw datasets in ../datasets/data/data_name/ directory.
# - file_names: mimic in this tutorial.

# In[ ]:


from datasets import CSVLoader

# Define data name
data_name = "practice"
# Define data dictionary
data_directory = "../datasets/data/practice/"

# Load train and test datasets
data_loader_training = CSVLoader(
    static_file=data_directory + "train_static.csv", temporal_file=data_directory + "train_temporal.csv"
)

data_loader_testing = CSVLoader(
    static_file=data_directory + "test_static.csv", temporal_file=data_directory + "test_temporal.csv"
)

dataset_training = data_loader_training.load()
dataset_testing = data_loader_testing.load()

print("Finish data loading.")


# ## Step 2: Preprocess dataset
#
# Preprocess the raw data using multiple filters. In this tutorial, we replace all the negative values to NaN (using NegativeFilter), do one-hot encoding on 'admission_type' feature (using OneHotEncoder), and do MinMax Normalization (using Normalization). Preprocessing is done for both training and testing datasets.
#   - NegativeFilter: Replace negative values to NaN
#   - OneHotEncoder: One hot encoding certain features
#     - one_hot_encoding: input features that need to be one-hot encoded
#   - Normalization (3 options): MinMax, Standard, None

# In[3]:


from preprocessing import FilterNegative, OneHotEncoder, Normalizer

# (1) filter out negative values
negative_filter = FilterNegative()

# (2) one-hot encode categorical features
one_hot_encoding = ["city", "state", "type"]
onehot_encoder = OneHotEncoder(one_hot_encoding_features=[one_hot_encoding])

# Data preprocessing
filter_pipeline = PipelineComposer(negative_filter, onehot_encoder)

dataset_training = filter_pipeline.fit_transform(dataset_training)
dataset_testing = filter_pipeline.transform(dataset_testing)

print("Finish preprocessing.")


# ## Step 3: Define problem
#
# Set the time-series prediction problem that we want to solve. Set the problem (whether it is one-shot or online prediction), set the label, set the maximum sequence length, and set the treatment features. We also define the metric for evaluation and the task itself (whether classification or regression). In this tutorial, we predict whether the patients will get ventilator after 4 hours (online setting).
#   - problem: 'one-shot'(one time prediction) or 'online'(rolling window prediction)
#     - 'one-shot': one time prediction at the end of the time-series
#     - 'online': preditcion at every time stamps of the time-series
#   - max_seq_len: maximum sequence length of time-series sequence
#   - label_name: the column name for the label(s)
#   - treatment: the column name for treatments
#   - window: x-hour ahead prediction.
#
#   - other parameters:
#     - metric_name: auc, apr, mse, mae
#     - task: classification or regression

# In[4]:

from preprocessing import ProblemMaker

# Define parameters
problem = "online"
max_seq_len = 24
label_name = "ventilator"
treatment = None
window = 4

# Define problem
problem_maker = ProblemMaker(
    problem=problem, label=[label_name], max_seq_len=max_seq_len, treatment=treatment, window=window
)

dataset_training = problem_maker.fit_transform(dataset_training)
dataset_testing = problem_maker.fit_transform(dataset_testing)

# Set other parameters
metric_name = "auc"
task = "classification"

metric_sets = [metric_name]
metric_parameters = {"problem": problem, "label_name": [label_name]}

print("Finish defining problem.")


# ## Step 4: Impute dataset
#
# Impute missing values in the preprocessed static and temporal datasets and return complete datasets.
#   - Static imputation (6 options): mean, median, mice, missforest, knn, gain
#   - Temporal imputation (8 options): mean, median, linear, quadratic, cubic, spline, mrnn, tgain

# In[5]:


from imputation import Imputation

# Set imputation models
static_imputation_model = "median"
temporal_imputation_model = "median"

# Impute the missing data
static_imputation = Imputation(imputation_model_name=static_imputation_model, data_type="static")
temporal_imputation = Imputation(imputation_model_name=temporal_imputation_model, data_type="temporal")

imputation_pipeline = PipelineComposer(static_imputation, temporal_imputation)

dataset_training = imputation_pipeline.fit_transform(dataset_training)
dataset_testing = imputation_pipeline.transform(dataset_testing)

print("Finish imputation.")


# ## Step 5: Feature selection
#
# Select the relevant static and temporal features to the labels. If you do not want, we can skip the feature selection (set feature selection method = None).
#   - feature selection method:
#       - feature_selection_model: greedy-addition, greedy-deletion, recursive-addition, recursive-deletion, None
#       - feature_number: selected featuer number

# In[6]:


from feature_selection import FeatureSelection

# Set feature selection parameters
static_feature_selection_model = None
temporal_feature_selection_model = None
static_feature_selection_number = None
temporal_feature_selection_number = None

# Select relevant features
static_feature_selection = FeatureSelection(
    feature_selection_model_name=static_feature_selection_model,
    feature_type="static",
    feature_number=static_feature_selection_number,
    task=task,
    metric_name=metric_name,
    metric_parameters=metric_parameters,
)

temporal_feature_selection = FeatureSelection(
    feature_selection_model_name=temporal_feature_selection_model,
    feature_type="temporal",
    feature_number=temporal_feature_selection_number,
    task=task,
    metric_name=metric_name,
    metric_parameters=metric_parameters,
)

feature_selection_pipeline = PipelineComposer(static_feature_selection, temporal_feature_selection)

dataset_training = feature_selection_pipeline.fit_transform(dataset_training)
dataset_testing = feature_selection_pipeline.transform(dataset_testing)

print("Finish feature selection.")


# ## Step 6: Time-series model fit and predict
#
# After finishing the data preparation, we define the predictive models (6 options, RNN, GRU, LSTM, Attention, Temporal CNN, and Transformer), and train the model using the training dataset. We set validation set as the 20% of the training set for early stopping and best model saving. After training, we use the trained model to predict the labels of the testing dataset.
#
# - predictor_parameters:
#   - model_name: rnn, gru, lstm, attention, tcn, transformer
#   - model_parameters: network parameters such as numer of layers
#     - h_dim: hidden dimensions
#     - n_layer: layer number
#     - n_head: head number (only for transformer model)
#     - batch_size: number of samples in mini-batch
#     - epochs: number of epochs
#     - learning_rate: learning rate
#   - static_mode: how to utilize static features (concatenate or None)
#   - time_mode: how to utilize time information (concatenate or None)

# In[7]:


from prediction import prediction

# Set predictive model
model_name = "gru"

# Set model parameters
model_parameters = {
    "h_dim": 100,
    "n_layer": 2,
    "n_head": 2,
    "batch_size": 128,
    "epoch": 20,
    "model_type": model_name,
    "learning_rate": 0.001,
    "static_mode": "Concatenate",
    "time_mode": "Concatenate",
    "verbose": True,
}

# Set up validation for early stopping and best model saving
dataset_training.train_val_test_split(prob_val=0.2, prob_test=0.0)

# Train the predictive model
pred_class = prediction(model_name, model_parameters, task)
pred_class.fit(dataset_training)
# Return the predictions on the testing set
test_y_hat = pred_class.predict(dataset_testing)

print("Finish predictor model training and testing.")


# ## Step 7: Estimate uncertainty
#
# Estimate uncertainty of the predictor models and returns the uncertainty of the predictions (test_ci_hat).
#
# - uncertainty_parameters:
#   - uncertainty estimation model name (ensemble)

# In[9]:


from uncertainty import uncertainty

# Set uncertainty model
uncertainty_model_name = "ensemble"

# Train uncertainty model
uncertainty_model = uncertainty(uncertainty_model_name, model_parameters, pred_class, task)
uncertainty_model.fit(dataset_training)
# Return uncertainty of the trained predictive model
test_ci_hat = uncertainty_model.predict(dataset_testing)

print("Finish uncertainty estimation")


# ## Step 8: Interpret predictions.
#
# Interpret the trained predictor model and return the instance-wise feature and temporal importance.
#
# - interpretor_parameters:
#   - interpretation_model_name: interpretation model name (tinvase)

# In[10]:


from interpretation import interpretation

# Set interpretation model
interpretation_model_name = "tinvase"

# Train interpretation model
interpretor = interpretation(interpretation_model_name, model_parameters, pred_class, task)
interpretor.fit(dataset_training)
# Return instance-wise temporal and static feature importance
test_s_hat = interpretor.predict(dataset_testing)

print("Finish model interpretation")


# ## Step 9: Visualize results
#
# (1) Visualize the performance of the trained model.

# In[11]:


from evaluation import Metrics
from evaluation import print_performance

# Evaluate predictor model
result = Metrics(metric_sets, metric_parameters).evaluate(dataset_testing.label, test_y_hat)
print("Finish predictor model evaluation.")

print("Overall performance")
print_performance(result, metric_sets, metric_parameters)


# ## Step 9: Visualize results
#
# (2) Visualize the predictions of a certain patient by trained predictive model.

# In[12]:


from evaluation import print_prediction

# Set the patient index for visualization
index = [1]

print("Each prediction")
print_prediction(test_y_hat[index], metric_parameters)


# ## Step 9: Visualize results
#
# (3) Visualize the uncertainty of a certain patient by trained predictive and uncertainty model.

# In[13]:


from evaluation import print_uncertainty

print("Uncertainty estimations")
print_uncertainty(test_y_hat[index], test_ci_hat[index], metric_parameters)


# ## Step 9: Visualize results
#
# (4) Visualize the interpretation of a certain patient prediction.

# In[14]:


from evaluation import print_interpretation

print("Model interpretation")
print_interpretation(test_s_hat[index], dataset_training.feature_name, metric_parameters, model_parameters)


# In[ ]:
