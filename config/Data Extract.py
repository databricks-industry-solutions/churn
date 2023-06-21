# Databricks notebook source
# MAGIC %md The purpose of this notebook is to download and set up the data we will use for the solution accelerator. Before running this notebook, make sure you have entered your own credentials for Kaggle and accepted the rules of this contest [dataset](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/rules).

# COMMAND ----------

# MAGIC %pip install kaggle py7zr

# COMMAND ----------

# MAGIC %md 
# MAGIC Set Kaggle credential configuration values in the block below: You can set up a [secret scope](https://docs.databricks.com/security/secrets/secret-scopes.html) to manage credentials used in notebooks. See the `./RUNME` notebook for a guide and script for setting up the `solution-accelerator-cicd` secret scope to saved credentials. 
# MAGIC
# MAGIC Don't forget to accept the [Terms of the challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data) before downloading data. 

# COMMAND ----------

import os
# os.environ['kaggle_username'] = 'YOUR KAGGLE USERNAME HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_username'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_username")

# os.environ['kaggle_key'] = 'YOUR KAGGLE KEY HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_key'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_key")

# COMMAND ----------

# MAGIC %md Download and unzip data:

# COMMAND ----------

# MAGIC %sh -e
# MAGIC cd /databricks/driver
# MAGIC export KAGGLE_USERNAME=$kaggle_username
# MAGIC export KAGGLE_KEY=$kaggle_key
# MAGIC kaggle competitions download -c kkbox-churn-prediction-challenge
# MAGIC unzip -o kkbox-churn-prediction-challenge.zip
# MAGIC py7zr x members_v3.csv.7z
# MAGIC py7zr x transactions.csv.7z
# MAGIC py7zr x transactions_v2.csv.7z
# MAGIC py7zr x user_logs.csv.7z
# MAGIC py7zr x user_logs_v2.csv.7z

# COMMAND ----------

# MAGIC %md Extract the downloaded data to the folder used throughout the accelerator:

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/members_v3.csv", "dbfs:/tmp/kkbox_churn/members/members_v3.csv")
dbutils.fs.mv("file:/databricks/driver/transactions.csv", "dbfs:/tmp/kkbox_churn/transactions/transactions.csv")
dbutils.fs.mv("file:/databricks/driver/data/churn_comp_refresh/transactions_v2.csv", "dbfs:/tmp/kkbox_churn/transactions/transactions_v2.csv")
dbutils.fs.mv("file:/databricks/driver/user_logs.csv", "dbfs:/tmp/kkbox_churn/user_logs/user_logs.csv")
dbutils.fs.mv("file:/databricks/driver/data/churn_comp_refresh/user_logs_v2.csv", "dbfs:/tmp/kkbox_churn/user_logs/user_logs_v2.csv")

# COMMAND ----------

# MAGIC %fs ls /tmp/kkbox_churn/transactions/

# COMMAND ----------


