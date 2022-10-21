# Databricks notebook source
# MAGIC %md The purpose of this notebook is to use our trained model to generate predictions that may be imported into a downstream CRM system.  
# MAGIC This series of notebook is also available at https://github.com/databricks-industry-solutions/churn. You can find more information about this accelerator at https://www.databricks.com/solutions/accelerators/retention-management.

# COMMAND ----------

# MAGIC %md ###Step 1: Retrieve Data for Scoring
# MAGIC 
# MAGIC The purpose of training our churn prediction model is to identify target customers for proactive retention management. As such, we need to periodically make predictions from feature information and make those predictions available within the systems supporting such campaigns.
# MAGIC 
# MAGIC With this in mind, we'll examine how we might retrieve our recently trained model and use it to generate scored output which can be imported into Salesforce, Microsoft Dynamics and many other systems accepting custom data imports.  While there are multiple paths for the integration of such output with these systems, we'll explore the simplest, *i.e.* a flat-file export.
# MAGIC 
# MAGIC To get started, we'll first retrieve feature data associated with the period for which we intend to make predictions.  Given we trained our model on February 2017 data and evaluated our model on March 2017 data, it would make sense for us to generate prediction output for April 2017.  That said, we want to avoid stepping on the toes of the Kaggle competition associated with this dataset so that we'll limit ourselves to generating March 2017 prediction output.
# MAGIC 
# MAGIC Unlike in previous notebooks, we'll limit data retrieval to features and a customer identifier, ignoring the churn lables as we would not have these if we were making actual future predictions. We'll load the data first into a Spark DataFrame and then into a pandas dataframe so that we might demonstrate two different techniques for generating output, each of which depends on a different dataframe type:

# COMMAND ----------

# DBTITLE 1,Import Needed Libraries
import mlflow
import mlflow.pyfunc

import pandas as pd
import shutil, os

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import struct

# COMMAND ----------

# DBTITLE 1,Set database for all the following queries
# please use a personalized database name here, if you wish to avoid interfering with other users who might be running this accelerator in the same workspace
database_name = 'kkbox_churn'
spark.sql(f'USE {database_name}')

# COMMAND ----------

# DBTITLE 1,Load Features & Customer Identifier
# retrieve features & identifier to Spark DataFrame
input = spark.sql('''
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
    b.ratio_number_total_after_exp_to_days_after_exp_with_session
  FROM test_trans_features a
  INNER JOIN test_act_features b
    ON a.msno=b.msno
  ''')

# extract features to pandas DataFrame
input_pd = input.toPandas()
X = input_pd.drop(['msno'], axis=1) # features for making predictions
msno = input_pd[['msno']] # customer identifiers to which we will append predictions

# COMMAND ----------

# MAGIC %md ###Step 2a: Generate Prediction Output Using the As-Is Model
# MAGIC 
# MAGIC Regardless of whether our intent is to use the [Microsoft Dynamics CRM Common Data Service](https://docs.microsoft.com/en-us/powerapps/developer/common-data-service/import-data) or [Salesforce DataLoader](https://developer.salesforce.com/docs/atlas.en-us.dataLoader.meta/dataLoader/data_loader.htm), we need to produce a UTF-8, delimited text file with a header row. Leveraging a pandas dataframe and the model's native functionality, we can deliver such a file as follows:

# COMMAND ----------

# DBTITLE 1,Retrieve Model from Registry
model_name = 'churn-ensemble'

model = mlflow.pyfunc.load_model(
  'models:/{0}/production'.format(model_name)
  )

# COMMAND ----------

# DBTITLE 1,Save Predictions to File
# databricks location for the output file
output_path = '/tmp/kkbox_churn/output_native/'
shutil.rmtree('/dbfs'+output_path, ignore_errors=True) # delete folder & contents if exists
dbutils.fs.mkdirs(output_path) # recreate folder

# generate predictions
y_prob = model.predict(X)

# assemble output dataset
output = pd.concat([
    msno, 
    pd.DataFrame(y_prob, columns=['churn'])
    ], axis=1
  )
output['period']='2017-03-01'

# write output to file
output[['msno', 'period', 'churn']].to_csv(
  path_or_buf='/dbfs'+output_path+'output.txt', # use /dbfs fuse mount to access cloud storage
  sep='\t',
  header=True,
  index=False,
  encoding='utf-8'
  )

# COMMAND ----------

# MAGIC %md And now we can examine the file and its contents:

# COMMAND ----------

# DBTITLE 1,Examine Output File Contents
print(
  dbutils.fs.head(output_path+'output.txt')
  )

# COMMAND ----------

# MAGIC %md ###Step 2b: Generate Prediction Output using Spark UDF
# MAGIC 
# MAGIC Using the native APIs on the sklearn model to make predictions (as was done in Step 2a) is familiar to most Data Scientists, but it's Data Engineers who will typically implementing this phase of our work.  For these individuals, using Spark SQL might be more familiar and would certainly scale better in scenarios when we are working with large numbers of customers.
# MAGIC 
# MAGIC To use our model within Spark, we must first register it as a user-defined function (UDF):

# COMMAND ----------

# DBTITLE 1,Register Model as UDF
churn_udf = mlflow.pyfunc.spark_udf(
  spark, 
  'models:/{0}/production'.format(model_name), 
  result_type = DoubleType()
  )

# COMMAND ----------

# MAGIC %md Now, we can use our UDF to generate predictions.  While it is possible to use the UDF within a SQL statement, because we have a very long list of features which we need to pass to the function, we will combine our feature fields using a Spark SQL *struct* type.  This will make passing a long-list of features easier and minimize future changes should our feature count increase:

# COMMAND ----------

# DBTITLE 1,Generate DataFrame with Predictions
output_path = '/tmp/kkbox_churn/output_spark/'

# get list of columns in dataframe
input_columns = input.columns

# assemble struct containing list of features
features = struct([feature_column for feature_column in input_columns if feature_column != 'msno']) 

# generate output dataset 
output = (
  input
    .withColumn(
      'churn', 
      churn_udf( features )  # call udf to generate prediction
      )
    .selectExpr('msno', '\'2017-03-01\' as period', 'churn')
  )

# write output to storage
(output
    .repartition(1)  # repartition to generate a single output file
    .write
    .mode('overwrite')
    .csv(
      path=output_path,
      sep='\t',
      header=True,
      encoding='UTF-8'
      )
  )

# COMMAND ----------

# MAGIC %md Notice in the cell above that we are writing our data to an output folder.  While we are able to use the *repartition()* method to ensure our data is written to a single file, we are not able to directly control the name of that file as it is generated here: 

# COMMAND ----------

# DBTITLE 1,Examine Output Folder Contents
display(
  dbutils.fs.ls(output_path)
       )

# COMMAND ----------

# MAGIC %md If naming the file something specific is important, we can use native Python functionality to rename the CSV output file after the fact:

# COMMAND ----------

# DBTITLE 1,Rename Output File
for file in os.listdir('/dbfs'+output_path):
  if file[-4:]=='.csv':
    shutil.move('/dbfs'+output_path+file, '/dbfs'+output_path+'output.txt' )

# COMMAND ----------

# MAGIC %md Examining the contents of this file, we can see it is identical to the content generated earlier, though the sort order of the customers may differ (as sorting was not specified in either step):

# COMMAND ----------

# DBTITLE 1,Examine Output File Contents
print(
  dbutils.fs.head(output_path+'output.txt')
  )
