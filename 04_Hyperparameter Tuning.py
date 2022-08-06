# Databricks notebook source
# MAGIC %md The purpose of this notebook is to tune the hyperparameters associated with our candidate models to arrive at an optimum configuration. 

# COMMAND ----------

# MAGIC %md ###Step 1: Load & Transform Data
# MAGIC 
# MAGIC To get started, we'll re-load our data, applying transformations to features to address issues related to missing data, categorical values & feature standardization.  This step is a repeat of work introduced and explained in the last notebook:

# COMMAND ----------

# DBTITLE 1,Import Needed Libraries
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

import numpy as np

import time

# COMMAND ----------

# DBTITLE 1,Set database for all the following queries
# please use a personalized database name here, if you wish to avoid interfering with other users who might be running this accelerator in the same workspace
database_name = 'kkbox_churn'
spark.sql(f'USE {database_name}')

# COMMAND ----------

# DBTITLE 1,Set up mlflow experiment in the user's personal workspace folder
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/churn"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

# DBTITLE 1,Load Features & Labels
# retreive training & testing data
train = spark.sql('''
  SELECT
    a.*,
    b.days_total,
    b.days_with_session,
    b.ratio_days_with_session_to_days,
    b.days_after_exp,
    b.days_after_exp_with_session,
    b.ratio_days_after_exp_with_session_to_days_after_exp,
    b.sessions_total,
    b.ratio_sessions_total_to_days_total,
    b.ratio_sessions_total_to_days_with_session,
    b.sessions_total_after_exp,
    b.ratio_sessions_total_after_exp_to_days_after_exp,
    b.ratio_sessions_total_after_exp_to_days_after_exp_with_session,
    b.seconds_total,
    b.ratio_seconds_total_to_days_total,
    b.ratio_seconds_total_to_days_with_session,
    b.seconds_total_after_exp,
    b.ratio_seconds_total_after_exp_to_days_after_exp,
    b.ratio_seconds_total_after_exp_to_days_after_exp_with_session,
    b.number_uniq,
    b.ratio_number_uniq_to_days_total,
    b.ratio_number_uniq_to_days_with_session,
    b.number_uniq_after_exp,
    b.ratio_number_uniq_after_exp_to_days_after_exp,
    b.ratio_number_uniq_after_exp_to_days_after_exp_with_session,
    b.number_total,
    b.ratio_number_total_to_days_total,
    b.ratio_number_total_to_days_with_session,
    b.number_total_after_exp,
    b.ratio_number_total_after_exp_to_days_after_exp,
    b.ratio_number_total_after_exp_to_days_after_exp_with_session,
    c.is_churn
  FROM train_trans_features a
  INNER JOIN train_act_features b
    ON a.msno=b.msno
  INNER JOIN train c
    ON a.msno=c.msno
  ''').toPandas()

test = spark.sql('''
  SELECT
    a.*,
    b.days_total,
    b.days_with_session,
    b.ratio_days_with_session_to_days,
    b.days_after_exp,
    b.days_after_exp_with_session,
    b.ratio_days_after_exp_with_session_to_days_after_exp,
    b.sessions_total,
    b.ratio_sessions_total_to_days_total,
    b.ratio_sessions_total_to_days_with_session,
    b.sessions_total_after_exp,
    b.ratio_sessions_total_after_exp_to_days_after_exp,
    b.ratio_sessions_total_after_exp_to_days_after_exp_with_session,
    b.seconds_total,
    b.ratio_seconds_total_to_days_total,
    b.ratio_seconds_total_to_days_with_session,
    b.seconds_total_after_exp,
    b.ratio_seconds_total_after_exp_to_days_after_exp,
    b.ratio_seconds_total_after_exp_to_days_after_exp_with_session,
    b.number_uniq,
    b.ratio_number_uniq_to_days_total,
    b.ratio_number_uniq_to_days_with_session,
    b.number_uniq_after_exp,
    b.ratio_number_uniq_after_exp_to_days_after_exp,
    b.ratio_number_uniq_after_exp_to_days_after_exp_with_session,
    b.number_total,
    b.ratio_number_total_to_days_total,
    b.ratio_number_total_to_days_with_session,
    b.number_total_after_exp,
    b.ratio_number_total_after_exp_to_days_after_exp,
    b.ratio_number_total_after_exp_to_days_after_exp_with_session,
    c.is_churn
  FROM test_trans_features a
  INNER JOIN test_act_features b
    ON a.msno=b.msno
  INNER JOIN test c
    ON a.msno=c.msno
  ''').toPandas()

# split into features and labels
X_train_raw = train.drop(['msno','is_churn'], axis=1)
y_train = train['is_churn']

X_test_raw = test.drop(['msno','is_churn'], axis=1)
y_test = test['is_churn']

# COMMAND ----------

# DBTITLE 1,Transform Features
# replace missing values
impute = ColumnTransformer(
  transformers=[('missing values', SimpleImputer(strategy='most_frequent'), ['last_payment_method', 'city', 'gender', 'registered_via', 'bd'])],
  remainder='passthrough'
  )

# encode categoricals and scale all others
encode_scale =  ColumnTransformer( 
  transformers= [('ohe categoricals', OneHotEncoder(categories='auto', drop='first'), slice(0,4))], # features 0 through 3 should be the first four features imputed in previous step
  remainder= StandardScaler()  # standardize all other features
  )

# package transformation logic
transform = Pipeline([
   ('impute', impute),
   ('encode_scale', encode_scale)
   ])

# transform data
X_train = transform.fit_transform(X_train_raw)
X_test = transform.transform(X_test_raw)

# COMMAND ----------

# MAGIC %md ###Step 2: Tune Hyperparameters (XGBClassifier)
# MAGIC 
# MAGIC The XGBClassifier makes available a [wide variety of hyperparameters](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier) which can be used to tune model training.  Using some knowledge of our data and the algorithm, we might attempt to manually set some of the hyperparameters. But given the complexity of the interactions between them, it can be difficult to know exactly which combination of values will provide us the best model results.  It's in scenarios such as these that we might perform a series of model runs with different hyperparameter settings to observe how the model responds and arrive at an optimal combination of values.
# MAGIC 
# MAGIC Using hyperopt, we can automate this task, providing the hyperopt framework with a range of potential values to explore.  Calling a function which trains the model and returns an evaluation metric, hyperopt can through the available search space to towards an optimum combination of values.
# MAGIC 
# MAGIC For model evaluation, we will be using the average precision (AP) score which increases towards 1.0 as the model improves.  Because hyperopt recognizes improvements as our evaluation metric declines, we will use -1 * the AP score as our loss metric within the framework. 
# MAGIC 
# MAGIC Putting this all together, we might arrive at model training and evaluation function as follows:

# COMMAND ----------

# DBTITLE 1,Define Model Evaluation Function for Hyperopt
def evaluate_model(hyperopt_params):
  
  # accesss replicated input data
  X_train_input = X_train_broadcast.value
  y_train_input = y_train_broadcast.value
  X_test_input = X_test_broadcast.value
  y_test_input = y_test_broadcast.value  
  
  # configure model parameters
  params = hyperopt_params
  
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
  # all other hyperparameters are taken as given by hyperopt
  
  #params['tree_method']='gpu_hist'      # settings for running on GPU
  #params['predictor']='gpu_predictor'   # settings for running on GPU
  
  params['tree_method']='hist'      # settings for running on CPU
  params['predictor']='cpu_predictor'   # settings for running on CPU
  
  # instantiate model with parameters
  model = XGBClassifier(**params)
  
  # train
  model.fit(X_train_input, y_train_input)
  
  # predict
  y_prob = model.predict_proba(X_test_input)
  
  # score
  model_ap = average_precision_score(y_test_input, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)  # record actual metric with mlflow run
  
  # invert metric for hyperopt
  loss = -1 * model_ap  
  
  # return results
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md The first part of the model evaluation function retrieves from memory replicated copies of our training and testing feature and label sets.  Our intent is to leverage SparkTrials in combination with hyperopt to parallelize the training of models across a Spark cluster, allowing us to perform multiple, simultaneous model training evaluation runs and reduce the overall time required to navigate the seach space.  By replicating our datasets to the worker nodes of the cluster, a task performed in the next cell, copies of the data needed for training and evaluation can be efficiently made available to the function with minimal networking overhead:
# MAGIC 
# MAGIC **NOTE** See the Distributed Hyperopt [best practices documentation](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-best-practices.html#handle-datasets-of-different-orders-of-magnitude-notebook) for more options for data distribution.

# COMMAND ----------

# DBTITLE 1,Replicate Training & Testing Datasets to Cluster Workers
X_train_broadcast = sc.broadcast(X_train)
X_test_broadcast = sc.broadcast(X_test)
y_train_broadcast = sc.broadcast(y_train)
y_test_broadcast = sc.broadcast(y_test)

# COMMAND ----------

# MAGIC %md The hyperparameter values delivered to the function by hyperopt are derived from a search space defined in the next cell.  Each hyperparameter in the search space is defined using an item in a dictionary, the name of which identifies the hyperparameter and the value of which defines a range of potential values for that parameter.  When defined using *hp.choice*, a parameter is selected from a predefined list of values.  When defined *hp.loguniform*, values are generated from a continuous range of values.  When defined using *hp.quniform*, values are generated from a continuous range but truncated to a level of precision identified by the third argument  in the range definition.  Hyperparameter search spaces in hyperopt may be defined in many other ways as indicated by the library's [online documentation](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions):  

# COMMAND ----------

# DBTITLE 1,Define Search Space
# define minimum positive class scale factor (as shown in previous notebook)
weights = compute_class_weight(
  'balanced', 
  classes=np.unique(y_train), 
  y=y_train
  )
scale = weights[1]/weights[0]

# define hyperopt search space
search_space = {
    'max_depth' : hp.quniform('max_depth', 1, 30, 1)                                  # depth of trees (preference is for shallow trees or even stumps (max_depth=1))
    ,'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.40))     # learning rate for XGBoost
    ,'gamma': hp.quniform('gamma', 0.0, 1.0, 0.001)                                   # minimum loss reduction required to make a further partition on a leaf node
    ,'min_child_weight' : hp.quniform('min_child_weight', 1, 20, 1)                   # minimum number of instances per node
    ,'subsample' : hp.loguniform('subsample', np.log(0.1), np.log(1.0))               # random selection of rows for training,
    ,'colsample_bytree' : hp.loguniform('colsample_bytree', np.log(0.1), np.log(1.0)) # proportion of columns to use per tree
    ,'colsample_bylevel': hp.loguniform('colsample_bylevel', np.log(0.1), np.log(1.0))# proportion of columns to use per level
    ,'colsample_bynode' : hp.loguniform('colsample_bynode', np.log(0.1), np.log(1.0)) # proportion of columns to use per node
    ,'scale_pos_weight' : hp.loguniform('scale_pos_weight', np.log(scale), np.log(scale * 10))   # weight to assign positive label to manage imbalance
    }

# COMMAND ----------

# MAGIC %md The remainder of the model evaluation function is fairly straightforward.  We simply train and evaluate our model and return our loss value, *i.e.* -1 * AP Score, as part of a dictionary interpretable by hyperopt.  Based on returned values, hyperopt will generate a new set of hyperparameter values from within the search space definition with which it will attempt to improve our metric. We will limit the number of hyperopt evaluations to 250 simply based on a few trail runs we performed (not shown).  The larger the potential search space and the degree to which the model (in combination with the training dataset) responds to different hyperparameter combinations determines how many iterations are required for hyperopt to arrive at locally optimal values.  You can examine the output of the hyperopt run to see how our loss metric slowly improves over the course of each of these evaluations:
# MAGIC 
# MAGIC **NOTE** The XGBClassifier is configured within the *evaluate_model* function to use **GPUs**. Make sure you are running this on a **GPU-based cluster**.

# COMMAND ----------

# perform evaluation
with mlflow.start_run(run_name='XGBClassifer'):
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
    max_evals=250,
    trials=SparkTrials(parallelism=4), # THIS VALUE IS ALIGNED WITH THE NUMBER OF WORKERS IN MY GPU-ENABLED CLUSTER (guidance differs for CPU-based clusters)
    verbose=True
    )

# COMMAND ----------

# MAGIC %md With our tuning exercise over, let's go ahead and release the replicated copies of our features and labels datasets.  This will take pressure off our cluster resources as we move forward:

# COMMAND ----------

# DBTITLE 1,Release Replicated Datasets
# release the broadcast datasets
X_train_broadcast.unpersist()
X_test_broadcast.unpersist()
y_train_broadcast.unpersist()
y_test_broadcast.unpersist()

# COMMAND ----------

# MAGIC %md Now we can examine the hyperparameter values arrived at by hyperopt:

# COMMAND ----------

# DBTITLE 1,Optimized Hyperparameter Settings
space_eval(search_space, argmin)

# COMMAND ----------

# MAGIC %md [Comparing the results](https://databricks.com/blog/2020/02/18/how-to-display-model-metrics-in-dashboards-using-the-mlflow-search-api.html) of different model runs automatically [captured in mlflow](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-spark-mlflow-integration.html) by hyperopt, we can see that some parameter settings have clear effects on our model's performance. Scrutinizing comparison charts can help us understand how hyperopt arrived at the settings above. 
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/churn_xgbclassifier_hyperparams.PNG">
# MAGIC 
# MAGIC Using our optimized hyperparameters, we can now train our model for [persistence](https://docs.databricks.com/applications/mlflow/models.html) in mlflow.  Notice that we are switching the tree method and predictor parameters to align with **CPU-based computation**. Later, we will package our models for deployment to a CPU-based infrastructure and this change aligns the model with that step. This should not affect the model output, only it's processing speed:
# MAGIC 
# MAGIC **NOTE** We are defining a list to hold the mlflow run id for this and other final model runs to enable easier retrieval later in the notebook.  This list will be used towards the end of this notebook to retrieve each of our persisted models for inclusion in a voting ensemble model.

# COMMAND ----------

# DBTITLE 1,Define Variable to Hold Final Model Info
# define list to hold run ids for later retrieval
run_ids = []

# COMMAND ----------

# DBTITLE 1,Train XGBClassifier Model
# train model with optimal settings 
with mlflow.start_run(run_name='XGB Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  run_name = run.data.tags['mlflow.runName']
  run_ids += [(run_name, run_id)]
   
  # configure params
  params = space_eval(search_space, argmin)
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])       
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight'])
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])
  if 'scale_pos_weight' in params: params['scale_pos_weight']=int(params['scale_pos_weight'])    
  params['tree_method']='hist'        # modified for CPU deployment
  params['predictor']='cpu_predictor' # modified for CPU deployment
  mlflow.log_params(params)
  
  # train
  model = XGBClassifier(**params)
  model.fit(X_train, y_train)
  mlflow.sklearn.log_model(model, 'model')  # persist model with mlflow
  
  # predict
  y_prob = model.predict_proba(X_test)
  
  # score
  model_ap = average_precision_score(y_test, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)
  
  print('Model logged under run_id "{0}" with AP score of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md ###Step 3: Train HistGradientBoostingClassifier & MLPClassifer Models
# MAGIC 
# MAGIC Using the same techniques as shown in the last step (but omitted here for brevity), we've identified an optimal set of parameters for the training of the HistGradientBoostingClassifier model.  We can now perform a final training run for this model:

# COMMAND ----------

# DBTITLE 1,Train HistGradientBoostingClassifier
# set optimal hyperparam values
params = {
 'learning_rate': 0.046117525858818814,
 'max_bins': 129.0,
 'max_depth': 7.0,
 'max_leaf_nodes': 44.0,
 'min_samples_leaf': 39.0,
 'scale_pos_factor': 1.4641157666623326
 }

# compute sample weights
sample_weights = compute_sample_weight(
  'balanced', 
  y=y_train
  )

# train model based on these hyper params
with mlflow.start_run(run_name='HGB Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  run_name = run.data.tags['mlflow.runName']
  run_ids += [(run_name, run_id)]
  
  # configure
  mlflow.log_params(params)
  params['max_depth'] = int(params['max_depth'])
  params['max_bins'] = int(params['max_bins'])
  params['min_samples_leaf'] = int(params['min_samples_leaf'])
  params['max_leaf_nodes'] = int(params['max_leaf_nodes'])
  sample_weights_factor = params.pop('scale_pos_factor')
  
  # train
  model = HistGradientBoostingClassifier(loss='binary_crossentropy', max_iter=1000, **params)
  model.fit(X_train, y_train, sample_weight = sample_weights * sample_weights_factor)
  mlflow.sklearn.log_model(model, 'model')
  
  # predict
  y_prob = model.predict_proba(X_test)
  
  # score
  model_ap = average_precision_score(y_test, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)
  
  print('Model logged under run_id "{0}" with AP Score of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md Having done the same for our neural network, we will train it now:

# COMMAND ----------

# DBTITLE 1,Train MLP Classifier
# optimal param settings
params = {
  'activation': 'logistic',
  'hidden_layer_1': 100.0,
  'hidden_layer_2': 35.0,
  'hidden_layer_cutoff': 15,
  'learning_rate': 'adaptive',
  'learning_rate_init': 0.3424456484117518,
  'solver': 'sgd'
   }

# train model based on these params
with mlflow.start_run(run_name='MLP Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  run_name = run.data.tags['mlflow.runName']
  run_ids += [(run_name, run_id)]
  
  mlflow.log_params(params)
  
  # hidden layer definitions
  hidden_layer_1 = int(params.pop('hidden_layer_1'))
  hidden_layer_2 = int(params.pop('hidden_layer_2'))
  hidden_layer_cutoff = int(params.pop('hidden_layer_cutoff'))
  if hidden_layer_2 > hidden_layer_cutoff:
    hidden_layer_sizes = (hidden_layer_1, hidden_layer_2)
  else:
    hidden_layer_sizes = (hidden_layer_1)
  params['hidden_layer_sizes']=hidden_layer_sizes
  
  # train
  model = MLPClassifier(max_iter=10000, **params)
  model.fit(X_train, y_train)
  mlflow.sklearn.log_model(model, 'model')
  
  # predict
  y_prob = model.predict_proba(X_test)
  
  # score
  model_ap = average_precision_score(y_test, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)
  
  print('Model logged under run_id "{0}" with AP Score of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md ###Step 4: Train Voting Classifier
# MAGIC 
# MAGIC We now have three optimized models, each of which has been persisted to mlflow. Using a voting ensemble, we might combine the predictions of these three models to produce a new prediction that's better than any one model on its own. To get us started, we'll need to retrieve the trained models from prior steps:

# COMMAND ----------

# DBTITLE 1,Retrieve Trained Models
models = []

# for each final training run, retreive its model from mlflow 
for run_id in run_ids:
  models += [(run_id[0], mlflow.sklearn.load_model('runs:/{0}/model'.format(run_id[1])))] 

models

# COMMAND ----------

# MAGIC %md We now need to consider how we want to combine the input of these models.  By default, each model receives equal consideration, but by adjusting the weighting assigned to each, we might be able to achieve a slightly better score.  If we define our weights as part of a search space, we can allow hyperopt to automate the examination of which weights give us the best results.  Notice that we are not defining our weights such that the combined weights add to one.  Instead, we'll allow the voting ensemble to perform the proportional weighting calculations for us:

# COMMAND ----------

# DBTITLE 1,Construct Search Space
search_space = {}

# for each model, define a weight hyperparameter
for i, model in enumerate(models):
    search_space['weight{0}'.format(i)] = hp.loguniform('weight{0}'.format(i), np.log(0.0001), np.log(1.000))

search_space

# COMMAND ----------

# MAGIC %md We now define our evaluation function as before.  Notice that in addition to accessing replicated versions of our datasets, we're accessing replicated versions of our pre-trained models through the models_broadcast variable which we will defined in a later step.  Broadcasting to the worker nodes does not need to be limited to datasets.
# MAGIC 
# MAGIC Because each model is pre-trained, there's no need for us to call the *fit()* method on the voting classifier. By-passing this call is important as fitting the HistGradientBoostingClassifier requires the passing of a sample_weights parameter which would be rejected by the other two models if we were to pass it through the *fit()* method as is required.  By-passing the method call is not explicitly supported by sklearn, but a [hack](https://stackoverflow.com/questions/42920148/using-sklearn-voting-ensemble-with-partial-fit) will allow us to trick the model to skip the step and still make predictions:

# COMMAND ----------

# DBTITLE 1,Define Ensemble Evaluation Function
def evaluate_model(hyperopt_params):
  
  # accesss replicated input data
  X_train_input = X_train_broadcast.value
  y_train_input = y_train_broadcast.value
  X_test_input = X_test_broadcast.value
  y_test_input = y_test_broadcast.value  
  models_input = models_broadcast.value  # pre-trained models
    
  # compile weights parameter used by the voting classifier (configured for 10 models max)
  weights = []
  for i in range(0,10):
    if 'weight{0}'.format(i) in hyperopt_params:
      weights += [hyperopt_params['weight{0}'.format(i)]]
    else:
      break
  
  # configure basic model
  model = VotingClassifier(
      estimators = models_input, 
      voting='soft',
      weights=weights
      )

  # configure model to recognize child models as pre-trained 
  clf_list = []
  for clf in models_input:
    clf_list += [clf[1]]
  model.estimators_ = clf_list
  
  # predict
  y_prob = model.predict_proba(X_test_input)
  
  # score
  model_ap = average_precision_score(y_test_input, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)
  
  # invert metric for hyperopt
  loss = -1 * model_ap  
  
  # return results
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md With our search space and model evaluation function defined, we can now iterate to find an optimal set of weights for the voting ensemble:

# COMMAND ----------

# DBTITLE 1,Optimize Ensemble Hyperparameters
X_train_broadcast = sc.broadcast(X_train)
X_test_broadcast = sc.broadcast(X_test)
y_train_broadcast = sc.broadcast(y_train)
y_test_broadcast = sc.broadcast(y_test)
models_broadcast = sc.broadcast(models)

# perform evalaution
with mlflow.start_run(run_name='Voting: {0}'.format('weights')):
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,
    max_evals=250,
    trials=SparkTrials(parallelism=4),
    verbose=True
    )
  
# release the broadcast dataset
X_train_broadcast.unpersist()
X_test_broadcast.unpersist()
y_train_broadcast.unpersist()
y_test_broadcast.unpersist()
models_broadcast.unpersist()

# COMMAND ----------

# DBTITLE 1,Print Optimized Hyperparameter Settings
space_eval(search_space, argmin)

# COMMAND ----------

# MAGIC %md Combining our three models, we are again able to achieve an AP score a bit better than we could with any model on its own.  Again, it's important to note that the weights arrived at through hyperopt are not the actual weightings applied to the individual models but a starting point for the calculation of proportional weights.  Because of this, using the mlflow comparison charts to examine how hyperopt arrives at these weights can be quite tricky.
# MAGIC 
# MAGIC But putting that aside, we can train and persist our final voting ensemble model for later re-use:

# COMMAND ----------

# DBTITLE 1,Train Ensemble Model
params = space_eval(search_space, argmin)

with mlflow.start_run(run_name='Voting Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  run_name = run.data.tags['mlflow.runName']
  run_ids += [(run_name, run_id)]
  
  mlflow.log_params(params)
  
  # compile weights (configured for 10 max)
  weights = []
  for i in range(0,10):
    if 'weight{0}'.format(i) in params:
      weights += [params['weight{0}'.format(i)]]
    else:
      break
  
  # configure basic model
  model = VotingClassifier(
      estimators = models, 
      voting='soft',
      weights=weights
      )

  # configure model to recognize child models are pre-trained 
  clf_list = []
  for clf in models:
    clf_list += [clf[1]]
  model.estimators_ = clf_list
  
  # predict
  y_prob = model.predict_proba(X_test)
  
  # score
  model_ap = average_precision_score(y_test, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)
  
  print('Model logged under run_id "{0}" with AP score of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md ###Step 5: Persist Model Pipeline
# MAGIC 
# MAGIC We now have an optimized model, trained and persisted.  However, it is built to expect pre-transformed data.  If we consider how we will use this model to make predictions for the business, it would be helpful to combine our data transformation steps, addressed at the top of this notebook, with our model so that untransformed feature data can be passed directly to it.  To tackle this, we'll take our ColumnTransformers defined earlier and our Voting Classifier model trained in the last step and combine them into a unified model pipeline:

# COMMAND ----------

# DBTITLE 1,Assemble Model Pipeline
# assemble pipeline
model_pipeline = Pipeline([
   ('impute', impute),
   ('encode_scale', encode_scale),
   ('voting_ensemble', model)
   ])

# COMMAND ----------

# MAGIC %md When defining a pipeline, we'd typically call the *fit()* method on it to train each of the steps, but each step has already been trained at various points in this notebook so that we can move directly into predictions.  To verify our pipeline works as expected, let's pass it our **raw** test data and calculate our evaluation metric to verify it is the same as observed in the last step:

# COMMAND ----------

# DBTITLE 1,Verify Model Pipeline Works 
# predict
y_prob = model_pipeline.predict_proba(X_test_raw)
  
# score
model_ap = average_precision_score(y_test, y_prob[:,1])

print('AP score: {0:.5f}'.format(model_ap))

# COMMAND ----------

# MAGIC %md Everything looks good.  We're just about ready to save this model for later reuse, but there's one last challenge we need to overcome.
# MAGIC 
# MAGIC We will be saving this model via mlflow and possibly registering it in Spark as a pandas UDF.  The default deployment of such a function in mlflow maps the pandas UDF to the model's *predict()* method.  As you may remember in our last notebook, the *predict()* method returns a class prediction of 0 or 1 based on a 50% probability threshold.  If we want to ensure our model returns the actual positive class probability when registered with any of the mlflow serving mechanisms, we'll need to write a customer wrapper that overrides the *predict()* method:

# COMMAND ----------

# DBTITLE 1,Define Wrapper to Override Predict Method
# shamelessly stolen from https://docs.databricks.com/_static/notebooks/mlflow/mlflow-end-to-end-example-aws.html

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# COMMAND ----------

# MAGIC %md Now we can persist our model, making sure we include our custom wrapper in the call:

# COMMAND ----------

# DBTITLE 1,Persist Model Pipeline 
with mlflow.start_run(run_name='Final Pipeline Model') as run:
  
  run_id = run.info.run_id
  
  # record the score with this model
  mlflow.log_metric('avg precision', model_ap)
  
  # persist the model with the custom wrapper
  wrappedModel = SklearnModelWrapper(model_pipeline)
  mlflow.pyfunc.log_model(
    artifact_path='model', 
    python_model=wrappedModel
    )
  
print('Model logged under run_id "{0}" with log loss of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md Now let's make this model our *production* instance using the [mlflow model registry](https://www.mlflow.org/docs/latest/model-registry.html) feature.  In a typical MLOps workflow, the movement of a model from initial registration to staging and then to production would involve multiple team members and a suite of tests to ensure production readiness.  For this demonstration, we'll move our model directly into production and archive any other production instances of it to ensure model retrieval in the next notebook is straightforward: 

# COMMAND ----------

model_name = 'churn-ensemble'

# archive any production model versions (from any previous runs of this notebook or manual workflow management)
client = mlflow.tracking.MlflowClient()
for mv in client.search_model_versions("name='{0}'".format(model_name)):
    # if model with this name is marked production
    if mv.current_stage.lower() == 'production':
      # mark is as archived
      client.transition_model_version_stage(
        name=mv.name,
        version=mv.version,
        stage='archived'
        )
      
# register last deployed model with mlflow model registry
mv = mlflow.register_model(
    'runs:/{0}/model'.format(run_id),
    'churn-ensemble'
    )
model_version = mv.version

# wait until newly registered model moves from PENDING_REGISTRATION to READY status
while mv.status == 'PENDING_REGISTRATION':
  time.sleep(5)
  for mv in client.search_model_versions("run_id='{0}'".format(run_id)):  # new search functionality in mlflow 1.10 will make easier
    if mv.version == model_version:
      break
      
# transition newly deployed model to production stage
client.transition_model_version_stage(
  name=model_name,
  version=model_version,
  stage='production'
  )      
