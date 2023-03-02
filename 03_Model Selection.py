# Databricks notebook source
# MAGIC %md The purpose of this notebook is to consider alternative models for churn prediction.  
# MAGIC This series of notebook is also available at https://github.com/databricks-industry-solutions/churn. You can find more information about this accelerator at https://www.databricks.com/solutions/accelerators/retention-management.

# COMMAND ----------

# MAGIC %md ###Step 1: Prepare Features & Labels
# MAGIC 
# MAGIC Our first step is to retrieve the features and labels with which we will train and evaluate our models.  The data preparation logic was examined in the last two notebooks:

# COMMAND ----------

# DBTITLE 1,Import Needed Libraries
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, log_loss, precision_recall_curve, auc, average_precision_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import numpy as np

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

# DBTITLE 1,Retrieve Features & Labels
# retrieve training dataset
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

# retrieve training dataset
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


# separate features and labels
X_train_raw = train.drop(['msno','is_churn'], axis=1)
y_train = train['is_churn']

# separate features and labels
X_test_raw = test.drop(['msno','is_churn'], axis=1)
y_test = test['is_churn']

# COMMAND ----------

# MAGIC %md Our feature sets, *X_train_raw* and *X_test_raw*, need to be transformed to address missing and categorical values.  In addition, we need to scale our continuous features to align them with the requirements of some of the models we will evaluate:

# COMMAND ----------

# DBTITLE 1,Transform the Feature Data
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

# apply transformations
X_train = transform.fit_transform(X_train_raw)
X_test = transform.transform(X_test_raw)

# COMMAND ----------

# MAGIC %md ###Step 2: Examine Evaluation Metrics
# MAGIC 
# MAGIC Churn prediction is typically addressed as a binary classification problem where a customer or subscription that has churned is identified as the *positive class*, *i.e.* assigned a churn label of 1, and a customer or subscription that has not churned is identified as the *negative class*, *i.e.* assigned a churn label of 0. In a well-run business, the number of churn events taking place in a given time period should be fairly low, creating an imbalance between the occurrence of negative and positive class events.  Another way of saying this is that the positive class is the *minority class* and the negative class is the *majority class*.  While great for the business, the imbalance between the minority and majority classes can create problems for the algorithms tasked with learning a predictive solution.
# MAGIC 
# MAGIC To understand this, let's examine the imbalance between negative and positive churn events in our training and testing datasets:

# COMMAND ----------

# DBTITLE 1,Examine the Balance between Class Labels
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   CONCAT(is_churn,' - ',CASE WHEN is_churn=0 THEN 'not churned' ELSE 'churned' END) as class,
# MAGIC   dataset,
# MAGIC   count(*) as instances
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     is_churn, 'train' as dataset
# MAGIC   FROM train
# MAGIC   UNION ALL
# MAGIC   SELECT
# MAGIC     is_churn, 'test' as dataset
# MAGIC   FROM test
# MAGIC   ) 
# MAGIC GROUP BY is_churn, dataset
# MAGIC ORDER BY is_churn DESC

# COMMAND ----------

# MAGIC %md With between 96 and 97% of our subscribers not churning in these two periods, we can achieve a very high accuracy score with a naive model that simply labels **every instance** as *not churning*:

# COMMAND ----------

# DBTITLE 1,Evaluate Naive Model Accuracy
# generate naive churn prediction of ALL negative class
naive_y_pred = np.zeros(y_test.shape[0], dtype='int32')

print('Naive model Accuracy:\t{0:.6f}'.format( accuracy_score(y_test, naive_y_pred)))

# COMMAND ----------

# MAGIC %md While we achieve what may look like a pretty decent score, we know this model does not align with our goal of identifying churning customers.  Let's take a look at how a trained model compares:

# COMMAND ----------

# DBTITLE 1,Train a Predictive Model
# train the model
trained_model = LogisticRegression(max_iter=1000)
trained_model.fit(X_train, y_train)

# predict
trained_y_pred = trained_model.predict(X_test)

# calculate accuracy
print('Trained Model Accuracy:\t{0:.6f}'.format(accuracy_score(y_test, trained_y_pred)))

# COMMAND ----------

# MAGIC %md In imbalanced scenarios, accuracy provides a very poor assessment of model performance.  Models with little to no skill can obtain high accuracy scores by simply leaning on majority class predictions. A better approach to model evaluation is to leverage metrics that examine the model's ability to predict the positive class as the positive class is considered to be of higher value or importance in a churn prediction scenario.  [Precision, recall and F1 scores](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) are commonly used for this kind of evaluation.
# MAGIC 
# MAGIC Still, metrics such as these don't tell the whole story. Each is calculated based on a predicted class assignment, *i.e.* 0 or 1.  Under the covers, the model is actually predicting a class probability, and those probabilities provide far more information than a simple class assignment:

# COMMAND ----------

# DBTITLE 1,Retrieve Trained Model Probabilities
# predict class assignment probabilities
trained_y_prob = trained_model.predict_proba(X_test)
trained_y_prob[110:120]

# COMMAND ----------

# MAGIC %md Similar probabilities might be assigned to our naive model using the proportion of each class in the training dataset:

# COMMAND ----------

# DBTITLE 1,Retrieve Naive Model Probabilities
# calculate ratio of negative class instances in training dataset
label_counts = np.unique(y_train, return_counts=True)[1]
negclass_prop = label_counts[0]/np.sum(label_counts)

# construct a set of class probabilities representing class proportions in training set
naive_y_prob = np.empty(trained_y_prob.shape, dtype='float')
naive_y_prob[:] = [negclass_prop, 1-negclass_prop]

# display results
naive_y_prob

# COMMAND ----------

# MAGIC %md To arrive at a class assignment, a threshold value is applied to the class probabilities. If one class or the other is above the threshold, that determines which class label is assigned to an instance. For the F1, precision and recall metrics mentioned previously, a threshold value must be set and the metric calculated based on label assignments.  The *predict()* method uses a 50% threshold but we could easily assign labels based on one of our choosing.
# MAGIC 
# MAGIC The more fundamental problem is that we don't know what an appropriate threshold is just yet.  And our threshold may vary by customer instance based on things like a customer's revenue/profit potential or other factors.  When our goal is to evaluate a model based on its predictive capacity over a range of potential threshold values, we might consider metrics that examine the prediction probabilities relative to the actual class labels.  One of the most popular of these is the ROC AUC score:
# MAGIC 
# MAGIC **NOTE** Only positive class probabilities are needed for this and similar metrics. The probability assigned to the positive class plus the probability assigned to the negative class will always equal 1.0.

# COMMAND ----------

# DBTITLE 1,Evaluate ROC AUC
# calculate ROC AUC for trained & naive models
trained_auc = roc_auc_score(y_test, trained_y_prob[:,1])
naive_auc = roc_auc_score(y_test, naive_y_prob[:,1])

print('Trained ROC AUC:\t{0:.6f}'.format(trained_auc))
print('Naive ROC AUC:\t\t{0:.6f}'.format(naive_auc))

# COMMAND ----------

# MAGIC %md The ROC AUC score measures the area under a receiver operator curve. The receiver operator curve (ROC) plots the shift in balance between true positive and false positive predictions as you increase the probability threshold with which you would identify an instance as a member of either the negative or positive class. Measuring the area addressed under this curve, we can identify the predictive capacity of our model across the complete range of threshold values.  The 0.50 score assigned to the naive model indicates it really has not predictive skill while the nearly 90% score assigned to the logistic regression model indicates it is far more certain of its predictions, even if the two models make very similar predictions at a 50% threshold (as reflected by their near identical accuracy scores).  Still, there is room for improvement as a perfect predictor can achieve a maximum ROC AUC score of 1.00.
# MAGIC 
# MAGIC Visualizing the curve (and the area under it) makes the concept of area under the curve (AUC) a little easier to understand:
# MAGIC 
# MAGIC **NOTE** In the chart below, the AUC for the naive model is shaded in purple while the AUC for the trained model encompasses both the red and purple shaded areas.

# COMMAND ----------

# DBTITLE 1,Visualize ROC AUC
trained_fpr, trained_tpr, trained_thresholds = roc_curve(y_test, trained_y_prob[:,1])
naive_fpr, naive_tpr, naive_thresholds = roc_curve(y_test, naive_y_prob[:,1])

# define the plot
fig, ax = plt.subplots(figsize=(10,8))

# plot the roc curve for the model
plt.plot(naive_fpr, naive_tpr, linestyle='--', label='Naive', color='xkcd:eggplant')
plt.plot(trained_fpr, trained_tpr, linestyle='solid', label='Trained', color='xkcd:cranberry')

# shade the area under the curve
ax.fill_between(trained_fpr, trained_tpr, 0, color='xkcd:light pink')
ax.fill_between(naive_fpr, naive_tpr, 0, color='xkcd:dusty lavender')

# label each curve with is ROC AUC score
ax.text(.55, .3, 'Naive AUC:  {0:.6f}'.format(naive_auc), fontsize=14)
ax.text(.32, .5, 'Trained AUC:  {0:.6f}'.format(trained_auc), fontsize=14)

# adjust the axes to intersect at (0,0) and (1,1)
ax.spines['left'].set_position(('data', 0.0))
ax.axes.spines['bottom'].set_position(('data', 0.0))
ax.axes.spines['right'].set_position(('data', 1.0))
ax.axes.spines['top'].set_position(('data', 1.0))

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# show the legend
plt.legend(loc=(0.05, 0.85))

# show the plot
plt.show()

# COMMAND ----------

# MAGIC %md Calculating the ROC AUC from the predicted probabilities provides us a way to examine differences between two models which at a given threshold may appear to make similar predictions. That said, many researchers have pointed out that in imbalanced scenarios (such as churn prediction), ROC AUC may not provide a reliable basis for model assessment. It is well worth reading [this white paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/) to understand why, but in a nutshell, the False Positive Rate plotted along the x-axis in an ROC is relatively insensitive owing to the low occurrence of positive class instances.  In other words, your model isn't likely to make a lot of false positive predictions if it's predictions are highly skewed towards the negative class.
# MAGIC 
# MAGIC As an alternative, we can examine a similar curve plotting precision and recall across a range of potential threshold values. Precision tells us what proportion of positive class predictions are correct, something we might describe as positive prediction accuracy.  Recall tells us what proportion of positive class instances in the dataset are identified by the model, giving us a measurement of the completeness of our predictions.  Plotting these two metrics across the range of potential thresholds yields a curve that's similar to the ROC curve but which tells us how our percentage of correct positive class predictions declines as we attempt to identify all positive classes. And also like the ROC curve, we can summarize the precision-recall curve (PRC) through and area under the curve calculation (PRC AUC).  
# MAGIC 
# MAGIC Still, AUC may be overly optimistic when computed against a PRC curve so that it is suggested an [average precision score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) (AP score) be used in its place.  The AP score calculates a weighted average precision across the range of thresholds with the change in recall between threshold values used as the weighting factor.  It provides a score that in some ways is similar to the AUC but tends to be a bit more conservative:

# COMMAND ----------

# DBTITLE 1,Evaluate AP Score
trained_ap = average_precision_score(y_test, trained_y_prob[:,1])
naive_ap = average_precision_score(y_test, naive_y_prob[:,1])

print('Naive AP:\t{0:.6f}'.format(naive_ap))
print('Trained AP:\t{0:.6f}'.format(trained_ap))

# COMMAND ----------

# MAGIC %md To help us better understand the relationship between precision and recall and how we might interpret a PRC AP score, let's examine a plot of precision and recall over a range of potential thresholds:

# COMMAND ----------

# DBTITLE 1,Visualize Precision-Recall
# get values for PR curve
naive_precision, naive_recall, naive_thresholds = precision_recall_curve(y_test, naive_y_prob[:,1])
naive_thresholds = np.append(naive_thresholds, 1)
trained_precision, trained_recall, trained_thresholds = precision_recall_curve(y_test, trained_y_prob[:,1])
trained_thresholds = np.append(trained_thresholds, 1)

# define the plot
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(32,10))

# precision
ax[0].set_title('Precision')
ax[0].plot(trained_thresholds, trained_precision, linestyle='solid', label='Trained', color='xkcd:grass green')
ax[0].fill_between(trained_thresholds, trained_precision, 0, color='xkcd:light grey green')

ax[0].spines['left'].set_position(('data', 0.0))
ax[0].spines['bottom'].set_position(('data', 0.0))
ax[0].spines['right'].set_position(('data', 1.0))
ax[0].spines['top'].set_position(('data', 1.0))

ax[0].set_xlabel('Threshold')
ax[0].set_ylabel('Precision')

# recall
ax[1].set_title('Recall')
ax[1].plot(trained_thresholds, trained_recall, linestyle='solid', label='Trained', color='xkcd:grass green')
ax[1].fill_between(trained_thresholds, trained_recall, 0, color='xkcd:light grey green')

ax[1].spines['left'].set_position(('data', 0.0))
ax[1].spines['bottom'].set_position(('data', 0.0))
ax[1].spines['right'].set_position(('data', 1.0))
ax[1].spines['top'].set_position(('data', 1.0))

ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('Recall')

# precision-recall curve

test_positive_prop = len(y_test[y_test==1]) / len(y_test)

ax[2].set_title('Precision-Recall')
ax[2].plot([0,1], [test_positive_prop,test_positive_prop], linestyle='--', label='Naive', color='xkcd:sunflower yellow')
ax[2].plot(trained_recall, trained_precision, linestyle='solid', label='Trained', color='xkcd:grass green')

# shade the area under the curve
ax[2].fill_between(trained_recall, trained_precision, 0, color='xkcd:light grey green')
ax[2].fill_between([0,1], [test_positive_prop,test_positive_prop], 0, color='xkcd:buff')

# label each curve with is ROC AUC score
ax[2].text(.2, .075, 'Naive AP:  {0:.6f}'.format(naive_ap), fontsize=14)
ax[2].text(.3, .3, 'Trained AP:  {0:.6f}'.format(trained_ap), fontsize=14)

# adjust the axes to intersect at (0,0) and (1,1)
ax[2].spines['left'].set_position(('data', 0.0))
ax[2].axes.spines['bottom'].set_position(('data', 0.0))
ax[2].axes.spines['right'].set_position(('data', 1.0))
ax[2].axes.spines['top'].set_position(('data', 1.0))

# axis labels
ax[2].set_xlabel('Recall')
ax[2].set_ylabel('Precision')

# show the legend
ax[2].legend(loc=(0.75, 0.85))

# COMMAND ----------

# MAGIC %md Starting with the precision curve on the far left-hand of the output, we can see that, in general, our model's ability to make accurate positive class predictions increases as we raise the threshold.  This is kind of like saying that the stronger the evidence, *i.e.* the higher the threshold requirement, the higher the percentage of correct predictions we'll make.  
# MAGIC 
# MAGIC So, what's happening with precision as we get around the 0.8 threshold?  The erratic nature of the chart above this threshold likely reflects a low number of positive class predictions with probabilities in this range.  We will tackle why our model struggles to predict positive class instances with greater certainty in the next step of this notebook.
# MAGIC 
# MAGIC And what about the naive model's precision (not plotted)? In our naive model, we state that every member has a positive class probability equivalent to the proportion of positive class members in the overall training dataset, about 3%.  Below a threshold of that same 3% value (~0.03) , we'd guess every instance is in the positive class but would only be correct about 4% of the time, roughly the proportion of positive class members in the testing set. Above that threshold, we'd fail to make any positive class predictions so that there'd be no precision metric to calculate.
# MAGIC 
# MAGIC Now we consider recall or the percentage of our actual positive class members we're able to identify through model prediction. If we set the threshold to 0, we can capture them all because we assume every instance is in the positive class.  Jumping back to the precision chart, you should see that only roughly 4% of those predictions would be correct. So while we'd identify all our positive class members at the zero threshold, we'd make a bunch of false positive predictions too. As the threshold is raised, we start missing more and more positive class instances until we are hardly capturing any positive class instances at all.
# MAGIC 
# MAGIC And the naive model? Below the ~0.03 threshold, the naive model would capture every positive instance because we assigned each instance a probability of roughly that value. But as soon as we cross that ~0.03 threshold, we're making no more positive class predictions and miss all positive class instances.
# MAGIC 
# MAGIC Considering how the precision and recall curves behave as thresholds move up and down, we can see there is a tug-of-war between these two metrics.  Move the threshold lower, you can capture more positive class instances which increases recall, but you'll also pick up a lot of incorrect predictions which lowers precision. Move the threshold higher, the accuracy of our positive class predictions increases though we are failing to predict more and more actual positive instances. 
# MAGIC 
# MAGIC This relationship is captured in the precision-recall curve (PRC). In an ideal situation, we predict each class instance with 100% certainty so that our precision is always 1.0 and our recall is 1.0 regardless of the threshold.  In the real world, our goal is to move our model increasingly closer to this ideal state, pushing the PRC towards the upper right-hand corner of the plot area.  As you can see from this chart, we have a lot of ground to cover before we start approaching this ideal. To summarize how far from this ideal we are, we can calculate an average precision (AP) score, which like AUC, will approach 1.00 as we move towards the ideal PRC. 
# MAGIC 
# MAGIC So, how does the naive model fit into the PRC? Remember that our naive model has about a 4% precision for thresholds at or below roughly 3% (~0.03).  (Again, these two values reflect the percentage of positive class members in the testing and training datasets, respectively.) Our model also has a 100% recall at thresholds at or below roughly 3% but then no recall as the threshold moves above this mark as it no longer identifies any positive instances. What this means for our PRC curve is that precision and recall have a fixed relationship in the naive scenario which we can illustrate with a horizontal bar set to the proportion of positive class instances in the testing class, about 4%.  This line does not vary as precision and recall have a fixed relationship at or below the identified threshold. 

# COMMAND ----------

# MAGIC %md ###Step 3: Explore Class Weightings & Log Loss
# MAGIC 
# MAGIC The AP score provides us a means to assess our model's overall ability to make positive class predictions.  We will use it as our primary evaluation metric.  That said, we need to examine one more critical metric: log loss.
# MAGIC 
# MAGIC Internally, many Machine Learning algorithms perform iterative optimization focused on minimizing model error.  Log loss is one popular error calculation used within Machine Learning algorithms.  In a nutshell, log loss calculates the gap between a class, *i.e.* 0 or 1, and the prediction probability associated with it.  The logarithmic-part of the metric refers to how the penalty exponentially grows as the probability moves further and further from an instances class assignment. 
# MAGIC 
# MAGIC By working to minimize log loss, the model is pushed to be more and more certain about its predictions.  But because there are a disproportionate number of negative class instances in the dataset, the model will receive the greatest reward if it learns to be more confident in its negative class predictions.  Uncertain positive class predictions impact the overall log loss score very little as there are simply far fewer positive class members in the set.
# MAGIC 
# MAGIC At a minimum, we need our model to put our negative and positive class instances on equal footing for the log loss calculation.  This can be done by multiplying the penalty associated with negative and positive class instances by a weight that reflects the proportion of the other class in the dataset.  This can be done by supplying the string 'balanced' to the *class_weight* argument of many ML model types:

# COMMAND ----------

# DBTITLE 1,Train Model with Equal Consideration of Positive and Negative Classes
# train the model
balanced_model = LogisticRegression(max_iter=10000, class_weight='balanced')
balanced_model.fit(X_train, y_train)

# predict
balanced_y_prob = balanced_model.predict_proba(X_test)

# score
balanced_ap = average_precision_score(y_test, balanced_y_prob[:,1])

# calculate accuracy
print('Trained AP:\t\t{0:.6f}'.format(trained_ap))
print('Balanced Model AP:\t{0:.6f}'.format(balanced_ap))

# COMMAND ----------

# MAGIC %md To more clearly see the weights used to balance the two classes, we can call the *compute_class_weight* utility which returns the same results as those generated by providing the 'balanced' value in the previous model run.  The first weight is assigned to the negative class and reduces its influence.  The second weight is assigned to the positive class and increases its influence:

# COMMAND ----------

weights = compute_class_weight(
  'balanced', 
  classes=np.unique(y_train), 
  y=y_train
  )

weights

# COMMAND ----------

# MAGIC %md It should be noted that a balanced weighting is only a suggested starting point.  This weighting puts the majority and minority class on equal footing and as the results of our last run show, this doesn't always translate into a better model score but instead a model that is equally good or bad at making positive and negative class predictions.
# MAGIC 
# MAGIC To better understand what balancing the class weights is doing, let's revisit the precision-recall curves from above:

# COMMAND ----------

# get values for PR curve
balanced_precision, balanced_recall, balanced_thresholds = precision_recall_curve(y_test, balanced_y_prob[:,1])
balanced_thresholds = np.append(balanced_thresholds, 1)


# define the plot
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(32,10))

# precision
ax[0].set_title('Precision')
ax[0].plot(trained_thresholds, trained_precision, linestyle='solid', label='Trained', color='xkcd:grass green')
ax[0].plot(balanced_thresholds, balanced_precision, linestyle='solid', label='Balanced', color='xkcd:blood orange')

ax[0].fill_between(trained_thresholds, trained_precision, 0, color='xkcd:light grey green')
ax[0].fill_between(balanced_thresholds, balanced_precision, 0, color='xkcd:pale salmon')

ax[0].spines['left'].set_position(('data', 0.0))
ax[0].spines['bottom'].set_position(('data', 0.0))
ax[0].spines['right'].set_position(('data', 1.0))
ax[0].spines['top'].set_position(('data', 1.0))

ax[0].set_xlabel('Threshold')
ax[0].set_ylabel('Precision')

# recall
ax[1].set_title('Recall')
ax[1].plot(trained_thresholds, trained_recall, linestyle='solid', label='Trained', color='xkcd:grass green')
ax[1].plot(balanced_thresholds, balanced_recall, linestyle='solid', label='Balanced', color='xkcd:blood orange')

ax[1].fill_between(trained_thresholds, trained_recall, 0, color='xkcd:light grey green')
ax[1].fill_between(balanced_thresholds, balanced_recall, 0, color='xkcd:pale salmon')

ax[1].spines['left'].set_position(('data', 0.0))
ax[1].spines['bottom'].set_position(('data', 0.0))
ax[1].spines['right'].set_position(('data', 1.0))
ax[1].spines['top'].set_position(('data', 1.0))

ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('Recall')

# precision-recall curve

ax[2].set_title('Precision-Recall')
ax[2].plot(trained_recall, trained_precision, linestyle='solid', label='Trained', color='xkcd:grass green')
ax[2].plot(balanced_recall, balanced_precision, linestyle='solid', label='Balanced', color='xkcd:blood orange')

# shade the area under the curve
ax[2].fill_between(trained_recall, trained_precision, 0, color='xkcd:light grey green')
ax[2].fill_between(balanced_recall, balanced_precision, 0, color='xkcd:pale salmon')

# adjust the axes to intersect at (0,0) and (1,1)
ax[2].spines['left'].set_position(('data', 0.0))
ax[2].axes.spines['bottom'].set_position(('data', 0.0))
ax[2].axes.spines['right'].set_position(('data', 1.0))
ax[2].axes.spines['top'].set_position(('data', 1.0))

# axis labels
ax[2].set_xlabel('Recall')
ax[2].set_ylabel('Precision')

# show the legend
ax[2].legend(loc=(0.75, 0.85))

# COMMAND ----------

# MAGIC %md The  curves very clearly show what's happening by applying balanced class weights. By putting the minority class on equal footing with the majority class, our positive class prediction accuracy (precision) increases more gradually as the threshold increases.  This would indicate our model is making a wider range of positive class predictions, not just predicting the "safe-bet" positives. The greater assertiveness of our model in terms of positive class prediction results in a greater number of our positive class instances being identified as illustrated by the recall chart.
# MAGIC 
# MAGIC Combined, the drop in precision with class balancing pushes the area under the PRC lower.  Still, it seems our model is poised to be a more reliable predictor of churn than before.
# MAGIC 
# MAGIC One last thing to note with regard to class weightings is that there's no need to apply a single weight to all members of a given class.  Instead, weights can be assigned at a per-instance level, a technique known as a sample weighting.  While not shown in these notebooks, such an approach would allow us to value different instances of the positive class based on metrics such as CLV.  Such an approach would allow us to train our model for  maximum profit retention. As [techniques evolve](https://www.sciencedirect.com/science/article/abs/pii/S0377221718310166), this may be an aspect of this exercise worth revisiting.

# COMMAND ----------

# MAGIC %md ###Step 4: Evaluate Different Algorithms
# MAGIC 
# MAGIC Another suggestion for dealing with class imbalance is to explore different algorithms to see if some may perform better than others in combination with a particular dataset.  Earlier, we used a Logistic Regression algorithm while  Random Forests, Gradient Boosted Trees and Neural Networks have all been shown to produce good results in these scenarios. To see how each behaves with our dataset, we'll train an instance of each, using mostly default parameter settings to see how they perform *out of the box*.  This is by no means an exhaustive evaluation of each model, but ideally it will point us towards one or more model types that may be a good match for our data:

# COMMAND ----------

# DBTITLE 1,Logistic Regression
# train the model
lreg_model = LogisticRegression(max_iter=10000, class_weight='balanced')
lreg_model.fit(X_train, y_train)

# predict
lreg_y_prob = lreg_model.predict_proba(X_test)

# evaluate
lreg_ap = average_precision_score(y_test, lreg_y_prob[:,1])

# COMMAND ----------

# DBTITLE 1,Random Forest
# train the model
rfc_model = RandomForestClassifier(class_weight='balanced')
rfc_model.fit(X_train, y_train)

# predict
rfc_y_prob = rfc_model.predict_proba(X_test)

# evaluate
rfc_ap = average_precision_score(y_test, rfc_y_prob[:,1])

# COMMAND ----------

# DBTITLE 1,Extreme Gradient Boosted Tree (XGBoost)
# normalize class weights so that positive class reflects a 1.0 weight on negative class
scale = weights[1]/weights[0]

# train the model
xgb_model = XGBClassifier(scale_pos_weight=scale) # similar to class_weights arg but applies to positive class only
xgb_model.fit(X_train, y_train)

# predict
xgb_y_prob = xgb_model.predict_proba(X_test)

# evaluate
xgb_ap = average_precision_score(y_test, xgb_y_prob[:,1])

# COMMAND ----------

# MAGIC %md **NOTE** The MLP Classifier does not support class or sample weighting.

# COMMAND ----------

# DBTITLE 1,Neural Network
# train the model
mlp_model = MLPClassifier(activation='relu', max_iter=1000)  # does not support class weighting
mlp_model.fit(X_train, y_train)

# predict
mlp_y_prob = mlp_model.predict_proba(X_test)

# evaluate
mlp_ap = average_precision_score(y_test, mlp_y_prob[:,1])

# COMMAND ----------

# MAGIC %md Let's now compare the evaluation metric for each model:

# COMMAND ----------

# DBTITLE 1,Compare Model Results
print('Logistic Regression AP:\t\t{0:.6f}'.format(lreg_ap))
print('RandomForest Classifier AP:\t{0:.6f}'.format(rfc_ap))
print('XGBoost Classifier AP:\t\t{0:.6f}'.format(xgb_ap))
print('MLP (Neural Network) AP:\t{0:.6f}'.format(mlp_ap))

# COMMAND ----------

# MAGIC %md Of our models, the XGBClassifier performed the best (followed closely by the neural network). This isn't terribly surprising given that XGBoost is featured heavily in many data classification competitions these days and is recognized as [relatively insensitive to class imbalances](https://www.sciencedirect.com/science/article/pii/S095741741101342X) such as are found in this particular dataset.
# MAGIC 
# MAGIC The XGBoost Classifier used above is but one of many gradient boosting classifiers available to us.  LightGBM is another popular of these model-types, and sklearn makes available the HistGradientBoostingClassifier which mimics its functionality:
# MAGIC 
# MAGIC **NOTE** The HistGradientBoostingClassifier doesn't support class weights so that we'll use sample weights with each instance in the set assigned weights which achieve the same effect.

# COMMAND ----------

# DBTITLE 1,Hist Gradient Boost Classifier
# compute sample weights (functionally equivalent to class weights when done in this manner)
sample_weights = compute_sample_weight(
  'balanced', 
  y=y_train
  )

# train the model
hgb_model = HistGradientBoostingClassifier(loss='binary_crossentropy', max_iter=1000)
hgb_model.fit(X_train, y_train, sample_weight=sample_weights)  # weighting applied to individual samples

# predict
hgb_y_prob = hgb_model.predict_proba(X_test)

# evaluate
hgb_ap = average_precision_score(y_test, hgb_y_prob[:,1])
print('HistGB Classifier AP:\t{0:.6f}'.format(hgb_ap))

# COMMAND ----------

# MAGIC %md The HistGradientBoostingClassifier did very well.  Still, it's difficult to say that any one of the models we trained here is truly *better* than the others based on the limited evaluation we've performed.  Instead, we should consider tuning each model to get it's best predictions and then see how the models compare before eliminating any from consideration.  Still, time is limited so that we'll go ahead and drop random forests and logistic regression from consideration moving forward as some limited testing on this dataset (along with information in the literature) would indicate they are not likely to provide us our best results.
# MAGIC 
# MAGIC One last model worth considering is the voting classifier.  This model type combines the predictions of multiple models to create an ensemble prediction. The *soft* voting setting instructs the model to average the probabilities generated by each of the models provided to it.  If some models performed more reliably than the others, we might apply weighting to the voting calculation to more highly favor those.  In the next notebook, we'll tackle model weighting, but for now, let's see how combining three models with equal weighting affects our evaluation metric:

# COMMAND ----------

# DBTITLE 1,Voting Ensemble
# train the model
vote_model = VotingClassifier(
  estimators=[
    ('hgb', HistGradientBoostingClassifier(loss='binary_crossentropy', max_iter=1000)), 
    ('xgb', XGBClassifier()),
    ('mlp', MLPClassifier(activation='relu', max_iter=1000))
    ],
  voting='soft'
  )
vote_model.fit(X_train, y_train)

# predict
vote_y_prob = vote_model.predict_proba(X_test)

# evaluate
vote_ap = average_precision_score(y_test, vote_y_prob[:,1])
print('Voting AP:\t{0:.6f}'.format(vote_ap))

# COMMAND ----------

# MAGIC %md Together, these models perform a little better together than they did on their own. After tuning the individual models (in the next notebook), we'll re-consider how a voting ensemble might be used to combine them.

# COMMAND ----------

# MAGIC %md ###Step 5: Consider Additional Options
# MAGIC 
# MAGIC Class imbalances are a particularly stubborn problem in generating reliable classification models. A number of additional strategies for dealing with imbalance have been identified and tend to fall into the categories of:</p>
# MAGIC 1. Modifying the algorithm
# MAGIC 2. Modifying the dataset
# MAGIC 
# MAGIC In the category of *modifying the algorithm*, we might (as we did above) consider different classes of models, some of which are less sensitive to class imbalance.  We might also look for opportunities to adjust the penalties (through class weights) used internally by different algorithms when performing iterative optimization to give higher consideration to the minority positive class. We've done a lightweight exploration of these techniques here, something we'll continue working on in the next notebook using our three top performing models plus the voting ensemble.
# MAGIC 
# MAGIC In the category of *modifying the dataset*, oversampling of the minority class, undersampling of the majority class, or a combination of the two techniques have been shown to help steer models towards more reliable predictions.  [These techniques](https://imbalanced-learn.readthedocs.io/en/stable/introduction.html) may select values from the dataset at random, use ML-guided approaches to identify values to select, or even use ML-guided techniques to synthesize new instances of minority class values or *correct* majority class values.
# MAGIC 
# MAGIC All that said, the ratio of negative to positive classes in our dataset is roughly 30:1. Most of the techniques in the *modify the dataset* category of techniques tend to be aimed at more highly imbalanced scenarios with class ratios of 100:1 or higher. Through some limited testing (not shown here) we determined that neither weighting nor alternative sampling techniques made significant improvements in our evaluation metrics. That's not to say that such techniques could not make improvements in a 30:1 or lower class imbalance.  One consistent theme in the literature surrounding these techniques is that no one technique is a solve for all imbalance problems and results vary significantly between datasets.
