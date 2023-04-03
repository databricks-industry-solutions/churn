# Databricks notebook source
# MAGIC %md The purpose of this notebook is to access & prepare the data required for churn prediction.  
# MAGIC This series of notebook is also available at https://github.com/databricks-industry-solutions/churn. You can find more information about this accelerator at https://www.databricks.com/solutions/accelerators/retention-management.

# COMMAND ----------

# MAGIC %md ###Step 1: Load the Data
# MAGIC 
# MAGIC In 2018, [KKBox](https://www.kkbox.com/), a popular music streaming service based in Taiwan, released a [dataset](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data) consisting of a little over two years of (anonymized) customer transaction and activity data with the goal of challenging the Data & AI community to predict which customers would churn in a future period.  
# MAGIC 
# MAGIC **NOTE** Due to the terms and conditions by which these data are made available, anyone interested in recreating this work will need to agree with the terms and conditions before making up this dataset and create a similar folder structure as described below in their environment. You can save the data permanently under a pre-defined [mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) named */mnt/kkbox*:
# MAGIC 
# MAGIC We have automated this downloading step for you and used a */tmp/kkbox_churn* storage path throughout this accelerator. 
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/kkbox_filedownloads.png' width=250>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Read into dataframes, these files form the following data model:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/kkbox_schema.png' width=300>
# MAGIC 
# MAGIC Each service subscriber is uniquely identified by a value in the *msno* field of the members table. Data in the transactions and user logs tables provide a record of subscription management and streaming activities, respectively.  Not every member has a complete set of data in this schema.  In addition, the transaction and streaming logs are quite verbose with multiple records being recorded for a subscriber on a given date.  On dates where there is no activity, no entries are found for a subscriber in these tables.
# MAGIC 
# MAGIC In order to protect data privacy, many values in these tables have been ordinal-encoded, limiting their interpretability.  In addition, timestamp information has been truncated to a daily level, making the sequencing of records on a given date dependent on business logic addressed in later steps in this notebook.
# MAGIC  
# MAGIC Let's load this data now:

# COMMAND ----------

# MAGIC %run "./config/Data Extract"

# COMMAND ----------

# DBTITLE 1,Import Libraries & Config Environment
import shutil
from datetime import date

from pyspark.sql.types import *
from pyspark.sql.functions import lit

# COMMAND ----------

# DBTITLE 1,Set to True to Skip Reload of Members, Transactions & User Logs Tables
# this has been added for scenarios where you might
# wish to alter some of the churn label prediction
# logic but do not wish to rerun the whole notebook
skip_reload = False

# please use a personalized database name here if you wish to avoid interfering with other users who might be running this accelerator in the same workspace
database_name = 'kkbox_churn'

# COMMAND ----------

# DBTITLE 1,Create Database to Contain Tables and Load Members Table
if skip_reload:
  # create database to house SQL tables
  _ = spark.sql(f'CREATE DATABASE IF NOT EXISTS {database_name}')
  _ = spark.sql(f'USE {database_name}')
else:
  # delete the old database if needed
  _ = spark.sql(f'DROP DATABASE IF EXISTS {database_name} CASCADE')
  _ = spark.sql(f'CREATE DATABASE {database_name}')
  _ = spark.sql(f'USE {database_name}')

  # drop any old delta lake files that might have been created
  shutil.rmtree('/dbfs/tmp/kkbox_churn/silver/members', ignore_errors=True)

  # members dataset schema
  member_schema = StructType([
    StructField('msno', StringType()),
    StructField('city', IntegerType()),
    StructField('bd', IntegerType()),
    StructField('gender', StringType()),
    StructField('registered_via', IntegerType()),
    StructField('registration_init_time', DateType())
    ])

  # read data from csv
  members = (
    spark
      .read
      .csv(
        '/tmp/kkbox_churn/members/members_v3.csv',
        schema=member_schema,
        header=True,
        dateFormat='yyyyMMdd'
        )
      )

  # persist in delta lake format
  (
    members
      .write
      .format('delta')
      .mode('overwrite')
      .save('/tmp/kkbox_churn/silver/members')
    )

    # create table object to make delta lake queryable
  _ = spark.sql('''
      CREATE TABLE members 
      USING DELTA 
      LOCATION '/tmp/kkbox_churn/silver/members'
      ''')

# COMMAND ----------

# DBTITLE 1,Load Transactions Table
if not skip_reload:

  # drop any old delta lake files that might have been created
  shutil.rmtree('/dbfs/tmp/kkbox_churn/silver/transactions', ignore_errors=True)

  # transaction dataset schema
  transaction_schema = StructType([
    StructField('msno', StringType()),
    StructField('payment_method_id', IntegerType()),
    StructField('payment_plan_days', IntegerType()),
    StructField('plan_list_price', IntegerType()),
    StructField('actual_amount_paid', IntegerType()),
    StructField('is_auto_renew', IntegerType()),
    StructField('transaction_date', DateType()),
    StructField('membership_expire_date', DateType()),
    StructField('is_cancel', IntegerType())  
    ])

  # read data from csv
  transactions = (
    spark
      .read
      .csv(
        '/tmp/kkbox_churn/transactions',
        schema=transaction_schema,
        header=True,
        dateFormat='yyyyMMdd'
        )
      )

  # persist in delta lake format
  ( transactions
      .write
      .format('delta')
      .partitionBy('transaction_date')
      .mode('overwrite')
      .save('/tmp/kkbox_churn/silver/transactions')
    )

    # create table object to make delta lake queryable
  _ = spark.sql('''
      CREATE TABLE transactions
      USING DELTA 
      LOCATION '/tmp/kkbox_churn/silver/transactions'
      ''')

# COMMAND ----------

# DBTITLE 1,Load User Logs Table
if not skip_reload:
  # drop any old delta lake files that might have been created
  shutil.rmtree('/dbfs/tmp/kkbox_churn/silver/user_logs', ignore_errors=True)

  # transaction dataset schema
  user_logs_schema = StructType([ 
    StructField('msno', StringType()),
    StructField('date', DateType()),
    StructField('num_25', IntegerType()),
    StructField('num_50', IntegerType()),
    StructField('num_75', IntegerType()),
    StructField('num_985', IntegerType()),
    StructField('num_100', IntegerType()),
    StructField('num_uniq', IntegerType()),
    StructField('total_secs', FloatType())  
    ])

  # read data from csv
  user_logs = (
    spark
      .read
      .csv(
        '/tmp/kkbox_churn/user_logs',
        schema=user_logs_schema,
        header=True,
        dateFormat='yyyyMMdd'
        )
      )

  # persist in delta lake format
  ( user_logs
      .write
      .format('delta')
      .partitionBy('date')
      .mode('overwrite')
      .save('/tmp/kkbox_churn/silver/user_logs')
    )

  # create table object to make delta lake queryable
  _ = spark.sql('''
    CREATE TABLE IF NOT EXISTS user_logs
    USING DELTA 
    LOCATION '/tmp/kkbox_churn/silver/user_logs'
    ''')

# COMMAND ----------

# MAGIC %md ###Step 2: Acquire Churn Labels
# MAGIC 
# MAGIC To build our model, we will need to identify which customers have churned within two periods of interest.  These periods are February 2017 and March 2017.  We will train our model to predict churn in February 2017 and then evaluate our model's ability to predict churn in March 2017, making these our training and testing datasets, respectively.
# MAGIC 
# MAGIC Per instructions provided in the Kaggle competition, a KKBox subscriber is not identified as churned until he or she fails to renew their subscription 30-days following its expiration.  Most subscriptions are themselves on a 30-day renewal schedule (though some subscriptions renew on significantly longer cycles). This means that identifying churn involves a sequential walk through the customer data, looking for renewal gaps that would indicate a customer churned on a prior expiration date.
# MAGIC 
# MAGIC While the competition makes available pre-labeled training and testing datasets, *train.csv* and *train_v2.csv*, respectively, several past participants have noted that these datasets should be regenerated.  A Scala script for doing so is provided by KKBox.  Modifying the script for this environment, we might regenerate our training and test datasets as follows:

# COMMAND ----------

# DBTITLE 1,Delete Training Labels (if exists)
_ = spark.sql('DROP TABLE IF EXISTS train')

shutil.rmtree('/dbfs/tmp/kkbox_churn/silver/train', ignore_errors=True)

# COMMAND ----------

# DBTITLE 1,Generate Training Labels (Logic Provided by KKBox)
# MAGIC %scala
# MAGIC  
# MAGIC import java.time.{LocalDate}
# MAGIC import java.time.format.DateTimeFormatter
# MAGIC import java.time.temporal.ChronoUnit
# MAGIC  
# MAGIC import org.apache.spark.sql.{Row, SparkSession}
# MAGIC import org.apache.spark.sql.functions._
# MAGIC import scala.collection.mutable
# MAGIC  
# MAGIC def calculateLastday(wrappedArray: mutable.WrappedArray[Row]) :String ={
# MAGIC   val orderedList = wrappedArray.sortWith((x:Row, y:Row) => {
# MAGIC     if(x.getAs[String]("transaction_date") != y.getAs[String]("transaction_date")) {
# MAGIC       x.getAs[String]("transaction_date") < y.getAs[String]("transaction_date")
# MAGIC     } else {
# MAGIC       
# MAGIC       val x_sig = x.getAs[String]("plan_list_price") +
# MAGIC         x.getAs[String]("payment_plan_days") +
# MAGIC         x.getAs[String]("payment_method_id")
# MAGIC  
# MAGIC       val y_sig = y.getAs[String]("plan_list_price") +
# MAGIC         y.getAs[String]("payment_plan_days") +
# MAGIC         y.getAs[String]("payment_method_id")
# MAGIC  
# MAGIC       //same plan, always subscribe then unsubscribe
# MAGIC       if(x_sig != y_sig) {
# MAGIC         x_sig > y_sig
# MAGIC       } else {
# MAGIC         if(x.getAs[String]("is_cancel")== "1" && y.getAs[String]("is_cancel") == "1") {
# MAGIC           //multiple cancel, consecutive cancels should only put the expiration date earlier
# MAGIC           x.getAs[String]("membership_expire_date") > y.getAs[String]("membership_expire_date")
# MAGIC         } else if(x.getAs[String]("is_cancel")== "0" && y.getAs[String]("is_cancel") == "0") {
# MAGIC           //multiple renewal, expiration date keeps extending
# MAGIC           x.getAs[String]("membership_expire_date") < y.getAs[String]("membership_expire_date")
# MAGIC         } else {
# MAGIC           //same day same plan transaction: subscription precedes cancellation
# MAGIC           x.getAs[String]("is_cancel") < y.getAs[String]("is_cancel")
# MAGIC         }
# MAGIC       }
# MAGIC     }
# MAGIC   })
# MAGIC   orderedList.last.getAs[String]("membership_expire_date")
# MAGIC }
# MAGIC  
# MAGIC def calculateRenewalGap(log:mutable.WrappedArray[Row], lastExpiration: String): Int = {
# MAGIC   val orderedDates = log.sortWith((x:Row, y:Row) => {
# MAGIC     if(x.getAs[String]("transaction_date") != y.getAs[String]("transaction_date")) {
# MAGIC       x.getAs[String]("transaction_date") < y.getAs[String]("transaction_date")
# MAGIC     } else {
# MAGIC       
# MAGIC       val x_sig = x.getAs[String]("plan_list_price") +
# MAGIC         x.getAs[String]("payment_plan_days") +
# MAGIC         x.getAs[String]("payment_method_id")
# MAGIC  
# MAGIC       val y_sig = y.getAs[String]("plan_list_price") +
# MAGIC         y.getAs[String]("payment_plan_days") +
# MAGIC         y.getAs[String]("payment_method_id")
# MAGIC  
# MAGIC       //same data same plan transaction, assumption: subscribe then unsubscribe
# MAGIC       if(x_sig != y_sig) {
# MAGIC         x_sig > y_sig
# MAGIC       } else {
# MAGIC         if(x.getAs[String]("is_cancel")== "1" && y.getAs[String]("is_cancel") == "1") {
# MAGIC           //multiple cancel of same plan, consecutive cancels should only put the expiration date earlier
# MAGIC           x.getAs[String]("membership_expire_date") > y.getAs[String]("membership_expire_date")
# MAGIC         } else if(x.getAs[String]("is_cancel")== "0" && y.getAs[String]("is_cancel") == "0") {
# MAGIC           //multiple renewal, expire date keep extending
# MAGIC           x.getAs[String]("membership_expire_date") < y.getAs[String]("membership_expire_date")
# MAGIC         } else {
# MAGIC           //same date cancel should follow subscription
# MAGIC           x.getAs[String]("is_cancel") < y.getAs[String]("is_cancel")
# MAGIC         }
# MAGIC       }
# MAGIC     }
# MAGIC   })
# MAGIC  
# MAGIC   //Search for the first subscription after expiration
# MAGIC   //If active cancel is the first action, find the gap between the cancellation and renewal
# MAGIC   val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd")
# MAGIC   var lastExpireDate = LocalDate.parse(s"${lastExpiration.substring(0,4)}-${lastExpiration.substring(4,6)}-${lastExpiration.substring(6,8)}", formatter)
# MAGIC   var gap = 9999
# MAGIC   for(
# MAGIC     date <- orderedDates
# MAGIC     if gap == 9999
# MAGIC   ) {
# MAGIC     val transString = date.getAs[String]("transaction_date")
# MAGIC     val transDate = LocalDate.parse(s"${transString.substring(0,4)}-${transString.substring(4,6)}-${transString.substring(6,8)}", formatter)
# MAGIC     val expireString = date.getAs[String]("membership_expire_date")
# MAGIC     val expireDate = LocalDate.parse(s"${expireString.substring(0,4)}-${expireString.substring(4,6)}-${expireString.substring(6,8)}", formatter)
# MAGIC     val isCancel = date.getAs[String]("is_cancel")
# MAGIC  
# MAGIC     if(isCancel == "1") {
# MAGIC       if(expireDate.isBefore(lastExpireDate)) {
# MAGIC         lastExpireDate = expireDate
# MAGIC       }
# MAGIC     } else {
# MAGIC       gap = ChronoUnit.DAYS.between(lastExpireDate, transDate).toInt
# MAGIC     }
# MAGIC   }
# MAGIC   gap
# MAGIC }
# MAGIC  
# MAGIC val data = spark
# MAGIC   .read
# MAGIC   .option("header", value = true)
# MAGIC   .csv("/tmp/kkbox_churn/transactions/")
# MAGIC  
# MAGIC val historyCutoff = "20170131"
# MAGIC  
# MAGIC val historyData = data.filter(col("transaction_date")>="20170101" and col("transaction_date")<=lit(historyCutoff))
# MAGIC val futureData = data.filter(col("transaction_date") > lit(historyCutoff))
# MAGIC  
# MAGIC val calculateLastdayUDF = udf(calculateLastday _)
# MAGIC val userExpire = historyData
# MAGIC   .groupBy("msno")
# MAGIC   .agg(
# MAGIC     calculateLastdayUDF(
# MAGIC       collect_list(
# MAGIC         struct(
# MAGIC           col("payment_method_id"),
# MAGIC           col("payment_plan_days"),
# MAGIC           col("plan_list_price"),
# MAGIC           col("transaction_date"),
# MAGIC           col("membership_expire_date"),
# MAGIC           col("is_cancel")
# MAGIC         )
# MAGIC       )
# MAGIC     ).alias("last_expire")
# MAGIC   )
# MAGIC  
# MAGIC val predictionCandidates = userExpire
# MAGIC   .filter(
# MAGIC     col("last_expire") >= "20170201" and col("last_expire") <= "20170228"
# MAGIC   )
# MAGIC   .select("msno", "last_expire")
# MAGIC  
# MAGIC  
# MAGIC val joinedData = predictionCandidates
# MAGIC   .join(futureData,Seq("msno"), "left_outer")
# MAGIC  
# MAGIC val noActivity = joinedData
# MAGIC   .filter(col("payment_method_id").isNull)
# MAGIC   .withColumn("is_churn", lit(1))
# MAGIC  
# MAGIC  
# MAGIC val calculateRenewalGapUDF = udf(calculateRenewalGap _)
# MAGIC val renewals = joinedData
# MAGIC   .filter(col("payment_method_id").isNotNull)
# MAGIC   .groupBy("msno", "last_expire")
# MAGIC   .agg(
# MAGIC     calculateRenewalGapUDF(
# MAGIC       collect_list(
# MAGIC         struct(
# MAGIC           col("payment_method_id"),
# MAGIC           col("payment_plan_days"),
# MAGIC           col("plan_list_price"),
# MAGIC           col("transaction_date"),
# MAGIC           col("membership_expire_date"),
# MAGIC           col("is_cancel")
# MAGIC         )
# MAGIC       ),
# MAGIC       col("last_expire")
# MAGIC     ).alias("gap")
# MAGIC   )
# MAGIC  
# MAGIC val validRenewals = renewals.filter(col("gap") < 30)
# MAGIC   .withColumn("is_churn", lit(0))
# MAGIC val lateRenewals = renewals.filter(col("gap") >= 30)
# MAGIC   .withColumn("is_churn", lit(1))
# MAGIC  
# MAGIC val resultSet = validRenewals
# MAGIC   .select("msno","is_churn")
# MAGIC   .union(
# MAGIC     lateRenewals
# MAGIC       .select("msno","is_churn")
# MAGIC       .union(
# MAGIC         noActivity.select("msno","is_churn")
# MAGIC       )
# MAGIC   )
# MAGIC  
# MAGIC resultSet.write.format("delta").mode("overwrite").save("/tmp/kkbox_churn/silver/train/")

# COMMAND ----------

# DBTITLE 1,Access Training Labels
# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE train
# MAGIC USING DELTA
# MAGIC LOCATION '/tmp/kkbox_churn/silver/train/';
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM train;

# COMMAND ----------

# DBTITLE 1,Delete Testing Labels (if exists)
_ = spark.sql('DROP TABLE IF EXISTS test')

shutil.rmtree('/dbfs/tmp/kkbox_churn/silver/test', ignore_errors=True)

# COMMAND ----------

# DBTITLE 1,Generate Testing Labels (Logic Provided by KKBox)
# MAGIC %scala
# MAGIC 
# MAGIC import java.time.{LocalDate}
# MAGIC import java.time.format.DateTimeFormatter
# MAGIC import java.time.temporal.ChronoUnit
# MAGIC 
# MAGIC import org.apache.spark.sql.{Row, SparkSession}
# MAGIC import org.apache.spark.sql.functions._
# MAGIC import scala.collection.mutable
# MAGIC 
# MAGIC def calculateLastday(wrappedArray: mutable.WrappedArray[Row]) :String ={
# MAGIC   val orderedList = wrappedArray.sortWith((x:Row, y:Row) => {
# MAGIC     if(x.getAs[String]("transaction_date") != y.getAs[String]("transaction_date")) {
# MAGIC       x.getAs[String]("transaction_date") < y.getAs[String]("transaction_date")
# MAGIC     } else {
# MAGIC       val x_sig = x.getAs[String]("plan_list_price") +
# MAGIC         x.getAs[String]("payment_plan_days") +
# MAGIC         x.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC 
# MAGIC       val y_sig = y.getAs[String]("plan_list_price") +
# MAGIC         y.getAs[String]("payment_plan_days") +
# MAGIC         y.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC       //same plan, always subscribe then unsubscribe
# MAGIC       if(x_sig != y_sig) {
# MAGIC         x_sig > y_sig
# MAGIC       } else {
# MAGIC         if(x.getAs[String]("is_cancel")== "1" && y.getAs[String]("is_cancel") == "1") {
# MAGIC           //multiple cancel, consecutive cancels should only put the expiration date earlier
# MAGIC           x.getAs[String]("membership_expire_date") > y.getAs[String]("membership_expire_date")
# MAGIC         } else if(x.getAs[String]("is_cancel")== "0" && y.getAs[String]("is_cancel") == "0") {
# MAGIC           //multiple renewal, expiration date keeps extending
# MAGIC           x.getAs[String]("membership_expire_date") < y.getAs[String]("membership_expire_date")
# MAGIC         } else {
# MAGIC           //same day same plan transaction: subscription precedes cancellation
# MAGIC           x.getAs[String]("is_cancel") < y.getAs[String]("is_cancel")
# MAGIC         }
# MAGIC       }
# MAGIC     }
# MAGIC   })
# MAGIC   orderedList.last.getAs[String]("membership_expire_date")
# MAGIC }
# MAGIC 
# MAGIC def calculateRenewalGap(log:mutable.WrappedArray[Row], lastExpiration: String): Int = {
# MAGIC   val orderedDates = log.sortWith((x:Row, y:Row) => {
# MAGIC     if(x.getAs[String]("transaction_date") != y.getAs[String]("transaction_date")) {
# MAGIC       x.getAs[String]("transaction_date") < y.getAs[String]("transaction_date")
# MAGIC     } else {
# MAGIC       
# MAGIC       val x_sig = x.getAs[String]("plan_list_price") +
# MAGIC         x.getAs[String]("payment_plan_days") +
# MAGIC         x.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC       val y_sig = y.getAs[String]("plan_list_price") +
# MAGIC         y.getAs[String]("payment_plan_days") +
# MAGIC         y.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC       //same data same plan transaction, assumption: subscribe then unsubscribe
# MAGIC       if(x_sig != y_sig) {
# MAGIC         x_sig > y_sig
# MAGIC       } else {
# MAGIC         if(x.getAs[String]("is_cancel")== "1" && y.getAs[String]("is_cancel") == "1") {
# MAGIC           //multiple cancel of same plan, consecutive cancels should only put the expiration date earlier
# MAGIC           x.getAs[String]("membership_expire_date") > y.getAs[String]("membership_expire_date")
# MAGIC         } else if(x.getAs[String]("is_cancel")== "0" && y.getAs[String]("is_cancel") == "0") {
# MAGIC           //multiple renewal, expire date keep extending
# MAGIC           x.getAs[String]("membership_expire_date") < y.getAs[String]("membership_expire_date")
# MAGIC         } else {
# MAGIC           //same date cancel should follow subscription
# MAGIC           x.getAs[String]("is_cancel") < y.getAs[String]("is_cancel")
# MAGIC         }
# MAGIC       }
# MAGIC     }
# MAGIC   })
# MAGIC 
# MAGIC   //Search for the first subscription after expiration
# MAGIC   //If active cancel is the first action, find the gap between the cancellation and renewal
# MAGIC   val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd")
# MAGIC   var lastExpireDate = LocalDate.parse(s"${lastExpiration.substring(0,4)}-${lastExpiration.substring(4,6)}-${lastExpiration.substring(6,8)}", formatter)
# MAGIC   var gap = 9999
# MAGIC   for(
# MAGIC     date <- orderedDates
# MAGIC     if gap == 9999
# MAGIC   ) {
# MAGIC     val transString = date.getAs[String]("transaction_date")
# MAGIC     val transDate = LocalDate.parse(s"${transString.substring(0,4)}-${transString.substring(4,6)}-${transString.substring(6,8)}", formatter)
# MAGIC     val expireString = date.getAs[String]("membership_expire_date")
# MAGIC     val expireDate = LocalDate.parse(s"${expireString.substring(0,4)}-${expireString.substring(4,6)}-${expireString.substring(6,8)}", formatter)
# MAGIC     val isCancel = date.getAs[String]("is_cancel")
# MAGIC 
# MAGIC     if(isCancel == "1") {
# MAGIC       if(expireDate.isBefore(lastExpireDate)) {
# MAGIC         lastExpireDate = expireDate
# MAGIC       }
# MAGIC     } else {
# MAGIC       gap = ChronoUnit.DAYS.between(lastExpireDate, transDate).toInt
# MAGIC     }
# MAGIC   }
# MAGIC   gap
# MAGIC }
# MAGIC 
# MAGIC val data = spark
# MAGIC   .read
# MAGIC   .option("header", value = true)
# MAGIC   .csv("/tmp/kkbox_churn/transactions/")
# MAGIC 
# MAGIC val historyCutoff = "20170228"
# MAGIC 
# MAGIC val historyData = data.filter(col("transaction_date")>="20170201" and col("transaction_date")<=lit(historyCutoff))
# MAGIC val futureData = data.filter(col("transaction_date") > lit(historyCutoff))
# MAGIC 
# MAGIC val calculateLastdayUDF = udf(calculateLastday _)
# MAGIC val userExpire = historyData
# MAGIC   .groupBy("msno")
# MAGIC   .agg(
# MAGIC     calculateLastdayUDF(
# MAGIC       collect_list(
# MAGIC         struct(
# MAGIC           col("payment_method_id"),
# MAGIC           col("payment_plan_days"),
# MAGIC           col("plan_list_price"),
# MAGIC           col("transaction_date"),
# MAGIC           col("membership_expire_date"),
# MAGIC           col("is_cancel")
# MAGIC         )
# MAGIC       )
# MAGIC     ).alias("last_expire")
# MAGIC   )
# MAGIC 
# MAGIC val predictionCandidates = userExpire
# MAGIC   .filter(
# MAGIC     col("last_expire") >= "20170301" and col("last_expire") <= "20170331"
# MAGIC   )
# MAGIC   .select("msno", "last_expire")
# MAGIC 
# MAGIC 
# MAGIC val joinedData = predictionCandidates
# MAGIC   .join(futureData,Seq("msno"), "left_outer")
# MAGIC 
# MAGIC val noActivity = joinedData
# MAGIC   .filter(col("payment_method_id").isNull)
# MAGIC   .withColumn("is_churn", lit(1))
# MAGIC 
# MAGIC 
# MAGIC val calculateRenewalGapUDF = udf(calculateRenewalGap _)
# MAGIC val renewals = joinedData
# MAGIC   .filter(col("payment_method_id").isNotNull)
# MAGIC   .groupBy("msno", "last_expire")
# MAGIC   .agg(
# MAGIC     calculateRenewalGapUDF(
# MAGIC       collect_list(
# MAGIC         struct(
# MAGIC           col("payment_method_id"),
# MAGIC           col("payment_plan_days"),
# MAGIC           col("plan_list_price"),
# MAGIC           col("transaction_date"),
# MAGIC           col("membership_expire_date"),
# MAGIC           col("is_cancel")
# MAGIC         )
# MAGIC       ),
# MAGIC       col("last_expire")
# MAGIC     ).alias("gap")
# MAGIC   )
# MAGIC 
# MAGIC val validRenewals = renewals.filter(col("gap") < 30)
# MAGIC   .withColumn("is_churn", lit(0))
# MAGIC val lateRenewals = renewals.filter(col("gap") >= 30)
# MAGIC   .withColumn("is_churn", lit(1))
# MAGIC 
# MAGIC val resultSet = validRenewals
# MAGIC   .select("msno","is_churn")
# MAGIC   .union(
# MAGIC     lateRenewals
# MAGIC       .select("msno","is_churn")
# MAGIC       .union(
# MAGIC         noActivity.select("msno","is_churn")
# MAGIC       )
# MAGIC   )
# MAGIC 
# MAGIC resultSet.write.format("delta").mode("overwrite").save("/tmp/kkbox_churn/silver/test/")

# COMMAND ----------

# DBTITLE 1,Access Testing Labels
# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE test
# MAGIC USING DELTA
# MAGIC LOCATION '/tmp/kkbox_churn/silver/test/';
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM test;

# COMMAND ----------

# MAGIC %md ###Step 3: Cleanse & Enhance Transaction Logs
# MAGIC 
# MAGIC In the churn script provided by KKBox (and used in the last step), time between transaction events is used in order to determine churn status. In situations where multiple transactions are recorded on a given date, complex logic is used to determine which transaction represents the final state of the account on that date.  This logic states that when we have multiple transactions for a given subscriber on a given date, we should:</p>
# MAGIC 
# MAGIC 1. Concatenate the plan_list_price, payment_plan_days, and payment_method_id values and consider the "bigger" of these values as preceding the others<br>
# MAGIC 2. If the concatenated value (defined in the last step) is the same across records for this date, cancellations, *i.e.* records where is_cancel=1, should follow other transactions<br>
# MAGIC 3. If there are multiple cancellations in this sequence, the record with the earliest expiration date is the last record for this transaction date<br>
# MAGIC 4. If there are no cancellations but multiple non-cancellations in this sequence, the non-cancellation record with the latest expiration date is the last record on the transaction date<br>
# MAGIC 
# MAGIC Rewriting this logic in SQL allows us to generate a cleansed version of the transaction log with the final record for each date:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS transactions_clean;
# MAGIC 
# MAGIC CREATE TABLE transactions_clean
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   WITH 
# MAGIC     transaction_sequenced (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         transaction_date,
# MAGIC         plan_list_price,
# MAGIC         payment_plan_days,
# MAGIC         payment_method_id,
# MAGIC         is_cancel,
# MAGIC         membership_expire_date,
# MAGIC         RANK() OVER (PARTITION BY msno, transaction_date ORDER BY plan_sort DESC, is_cancel) as sort_id  -- calc rank on price, days & method sort followed by cancel sort
# MAGIC       FROM (
# MAGIC         SELECT
# MAGIC           msno,
# MAGIC           transaction_date,
# MAGIC           plan_list_price,
# MAGIC           payment_plan_days,
# MAGIC           payment_method_id,
# MAGIC           CONCAT(CAST(plan_list_price as string), CAST(payment_plan_days as string), CAST(payment_method_id as string)) as plan_sort,
# MAGIC           is_cancel,
# MAGIC           membership_expire_date
# MAGIC         FROM transactions
# MAGIC         )
# MAGIC       )
# MAGIC   SELECT
# MAGIC     p.msno,
# MAGIC     p.transaction_date,
# MAGIC     p.plan_list_price,
# MAGIC     p.actual_amount_paid,
# MAGIC     p.plan_list_price - p.actual_amount_paid as discount,
# MAGIC     p.payment_plan_days,
# MAGIC     p.payment_method_id,
# MAGIC     p.is_cancel,
# MAGIC     p.is_auto_renew,
# MAGIC     p.membership_expire_date
# MAGIC   FROM transactions p
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       x.msno,
# MAGIC       x.transaction_date,
# MAGIC       x.plan_list_price,
# MAGIC       x.payment_plan_days,
# MAGIC       x.payment_method_id,
# MAGIC       x.is_cancel,
# MAGIC       CASE   -- if is_cancel is 0 in last record then go with max membership date identified, otherwise go with lowest membership date
# MAGIC         WHEN x.is_cancel=0 THEN MAX(x.membership_expire_date)
# MAGIC         ELSE MIN(x.membership_expire_date)
# MAGIC         END as membership_expire_date
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         a.msno,
# MAGIC         a.transaction_date,
# MAGIC         a.plan_list_price,
# MAGIC         a.payment_plan_days,
# MAGIC         a.payment_method_id,
# MAGIC         a.is_cancel,
# MAGIC         a.membership_expire_date
# MAGIC       FROM transaction_sequenced a
# MAGIC       INNER JOIN (
# MAGIC         SELECT msno, transaction_date, MAX(sort_id) as max_sort_id -- find last entries on a given date
# MAGIC         FROM transaction_sequenced 
# MAGIC         GROUP BY msno, transaction_date
# MAGIC         ) b
# MAGIC         ON a.msno=b.msno AND a.transaction_date=b.transaction_date AND a.sort_id=b.max_sort_id
# MAGIC         ) x
# MAGIC     GROUP BY 
# MAGIC       x.msno, 
# MAGIC       x.transaction_date, 
# MAGIC       x.plan_list_price,
# MAGIC       x.payment_plan_days,
# MAGIC       x.payment_method_id,
# MAGIC       x.is_cancel
# MAGIC    ) q
# MAGIC    ON 
# MAGIC      p.msno=q.msno AND 
# MAGIC      p.transaction_date=q.transaction_date AND 
# MAGIC      p.plan_list_price=q.plan_list_price AND 
# MAGIC      p.payment_plan_days=q.payment_plan_days AND 
# MAGIC      p.payment_method_id=q.payment_method_id AND 
# MAGIC      p.is_cancel=q.is_cancel AND 
# MAGIC      p.membership_expire_date=q.membership_expire_date;
# MAGIC      
# MAGIC SELECT * 
# MAGIC FROM transactions_clean
# MAGIC ORDER BY msno, transaction_date;

# COMMAND ----------

# MAGIC %md Using this *cleansed* transaction data, we can now more easily identify the start and end of subscriptions using the 30-day gap logic found in the Scala code.  It's important to note that over the 2+ year period represented by the dataset, many subscribers will churn and many of those that do churn will re-subscribe.  With this in mind, we will generate a subscription ID to identify the different subscriptions, each of which will have a non-overlapping starting and ending date for a given subscriber:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS subscription_windows;
# MAGIC 
# MAGIC CREATE TABLE subscription_windows 
# MAGIC USING delta
# MAGIC AS
# MAGIC   WITH end_dates (
# MAGIC       SELECT p.*
# MAGIC       FROM (
# MAGIC         SELECT
# MAGIC           m.msno,
# MAGIC           m.transaction_date,
# MAGIC           m.membership_expire_date,
# MAGIC           m.next_transaction_date,
# MAGIC           CASE
# MAGIC             WHEN m.next_transaction_date IS NULL THEN 1
# MAGIC             WHEN DATEDIFF(m.next_transaction_date, m.membership_expire_date) > 30 THEN 1
# MAGIC             ELSE 0
# MAGIC             END as end_flag,
# MAGIC           CASE
# MAGIC             WHEN m.next_transaction_date IS NULL THEN m.membership_expire_date
# MAGIC             WHEN DATEDIFF(m.next_transaction_date, m.membership_expire_date) > 30 THEN m.membership_expire_date
# MAGIC             ELSE DATE_ADD(m.next_transaction_date, -1)  -- then just move the needle to just prior to the next transaction
# MAGIC             END as end_date
# MAGIC         FROM (
# MAGIC           SELECT
# MAGIC             x.msno,
# MAGIC             x.transaction_date,
# MAGIC             CASE  -- correct backdated expirations for subscription end calculations
# MAGIC               WHEN x.membership_expire_date < x.transaction_date THEN x.transaction_date
# MAGIC               ELSE x.membership_expire_date
# MAGIC               END as membership_expire_date,
# MAGIC             LEAD(x.transaction_date, 1) OVER (PARTITION BY x.msno ORDER BY x.transaction_date) as next_transaction_date
# MAGIC           FROM transactions_clean x
# MAGIC           ) m
# MAGIC         ) p
# MAGIC       WHERE p.end_flag=1
# MAGIC     )
# MAGIC   SELECT
# MAGIC     ROW_NUMBER() OVER (ORDER BY subscription_start, msno) as subscription_id,
# MAGIC     msno,
# MAGIC     subscription_start,
# MAGIC     subscription_end
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       x.msno,
# MAGIC       MIN(x.transaction_date) as subscription_start,
# MAGIC       y.window_end as subscription_end
# MAGIC     FROM transactions_clean x
# MAGIC     INNER JOIN (
# MAGIC       SELECT
# MAGIC         a.msno,
# MAGIC         COALESCE( MAX(b.end_date), '2015-01-01') as window_start,
# MAGIC         a.end_date as window_end
# MAGIC       FROM end_dates a
# MAGIC       LEFT OUTER JOIN end_dates b
# MAGIC         ON a.msno=b.msno AND a.end_date > b.end_date
# MAGIC       GROUP BY a.msno, a.end_date
# MAGIC       ) y
# MAGIC       ON x.msno=y.msno AND x.transaction_date BETWEEN y.window_start AND y.window_end
# MAGIC     GROUP BY x.msno, y.window_end
# MAGIC     )
# MAGIC   ORDER BY subscription_id;
# MAGIC   
# MAGIC SELECT *
# MAGIC FROM subscription_windows
# MAGIC ORDER BY subscription_id;

# COMMAND ----------

# MAGIC %md To verify we have our subscription windows aligned with the script used to identify customers at-risk for churn in February and March 2017, let's perform a quick test.  The script identifies an at-risk subscription as one where the last transaction recorded in the historical period, *i.e.* the time period leading up to the start of the month of interest, has an expiration date falling between the 30-day window leading up to the start of the period of interest and the end of that period.  For example, if we were to identify at-risk customers for February 2017, we should look for those customers with active subscriptions set to expire within the 30-days before February 1, 2017 and February 28, 2017.  This shifted window allows time for the 30-day grace period to expire within the period of interest. 
# MAGIC 
# MAGIC **NOTE** Better logic would limit our assessment to those subscriptions with an expiration date between 30-days prior to the start of the period AND 30-days prior to the end of the period.  (Such logic would exclude subscriptions expiring within the period of interest but which do not exit the 30-day grace period until after the period is over.) When we use this logic, we find numerous subscriptions that the provided script identifies as at-risk but which we would not.  We will align our logic with that of the competition for this exercise.
# MAGIC 
# MAGIC With this logic in mind, let's see if all our labeled at-risk customers adhere to this logic:
# MAGIC 
# MAGIC **NOTE** The next two cells should return NO RESULTS if our logic is valid

# COMMAND ----------

# DBTITLE 1,Identify Any Subscriptions in Training Dataset Not Believed to Be At-Risk
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   x.msno
# MAGIC FROM train x
# MAGIC LEFT OUTER JOIN (
# MAGIC   SELECT DISTINCT -- subscriptions that had risk in Feb 2017
# MAGIC     a.msno
# MAGIC   FROM subscription_windows a
# MAGIC   INNER JOIN transactions_clean b
# MAGIC     ON a.msno=b.msno AND b.transaction_date BETWEEN a.subscription_start AND a.subscription_end
# MAGIC   WHERE 
# MAGIC         a.subscription_start < '2017-02-01' AND
# MAGIC         (b.membership_expire_date BETWEEN DATE_ADD('2017-02-01',-30) AND '2017-02-28')
# MAGIC   ) y
# MAGIC   ON x.msno=y.msno
# MAGIC WHERE y.msno IS NULL

# COMMAND ----------

# DBTITLE 1,Identify Any Subscriptions in Testing Dataset Not Believed to Be At-Risk
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   x.msno
# MAGIC FROM test x
# MAGIC LEFT OUTER JOIN (
# MAGIC   SELECT DISTINCT -- subscriptions that had risk in Feb 2017
# MAGIC     a.msno
# MAGIC   FROM subscription_windows a
# MAGIC   INNER JOIN transactions_clean b
# MAGIC     ON a.msno=b.msno AND b.transaction_date BETWEEN a.subscription_start AND a.subscription_end
# MAGIC   WHERE 
# MAGIC         a.subscription_start < '2017-03-01' AND
# MAGIC         (b.membership_expire_date BETWEEN DATE_ADD('2017-03-01',-30) AND '2017-03-31')
# MAGIC   ) y
# MAGIC   ON x.msno=y.msno
# MAGIC WHERE y.msno IS NULL

# COMMAND ----------

# MAGIC %md While we do not fail to identify the same at-risk subscriptions as the provided script, if we were to alter the code above we would find a few subscriptions that we do identify as at-risk but which the Scala script does not. While it might be useful to examine why this is, so long as there are no members that the Scala script identifies as at risk that we do not, we should should be able to use this dataset to derive features for subscriptions in our testing and training datasets.
# MAGIC 
# MAGIC Leveraging subscription duration information derived in the last few cells, we can now enhance our transaction log to detect account-level changes.  This information will form the basis for transaction-feature generation in the next notebook:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS transactions_enhanced;
# MAGIC 
# MAGIC CREATE TABLE transactions_enhanced
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   SELECT
# MAGIC     b.subscription_id,
# MAGIC     a.*,
# MAGIC     COALESCE( DATEDIFF(a.transaction_date, LAG(a.transaction_date, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date)), 0) as days_since_last_transaction,
# MAGIC     COALESCE( a.plan_list_price - LAG(a.plan_list_price, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date), 0) as change_in_list_price,
# MAGIC     COALESCE(a.actual_amount_paid - LAG(a.actual_amount_paid, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date), 0) as change_in_actual_amount_paid,
# MAGIC     COALESCE(a.discount - LAG(a.discount, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date), 0) as change_in_discount,
# MAGIC     COALESCE(a.payment_plan_days - LAG(a.payment_plan_days, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date), 0) as change_in_payment_plan_days,
# MAGIC     CASE WHEN (a.payment_method_id != LAG(a.payment_method_id, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date)) THEN 1 ELSE 0 END  as change_in_payment_method_id,
# MAGIC     CASE
# MAGIC       WHEN a.is_cancel = LAG(a.is_cancel, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date) THEN 0
# MAGIC       WHEN a.is_cancel = 0 THEN -1
# MAGIC       ELSE 1
# MAGIC       END as change_in_cancellation,
# MAGIC     CASE
# MAGIC       WHEN a.is_auto_renew = LAG(a.is_auto_renew, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date) THEN 0
# MAGIC       WHEN a.is_auto_renew = 0 THEN -1
# MAGIC       ELSE 1
# MAGIC       END as change_in_auto_renew,
# MAGIC     COALESCE( DATEDIFF(a.membership_expire_date, LAG(a.membership_expire_date, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date)), 0) as days_change_in_membership_expire_date
# MAGIC 
# MAGIC   FROM transactions_clean a
# MAGIC   INNER JOIN subscription_windows b
# MAGIC     ON a.msno=b.msno AND 
# MAGIC        a.transaction_date BETWEEN b.subscription_start AND b.subscription_end
# MAGIC   ORDER BY 
# MAGIC     a.msno,
# MAGIC     a.transaction_date;
# MAGIC     
# MAGIC SELECT * FROM transactions_enhanced;

# COMMAND ----------

# MAGIC %md ###Step 4: Generate Dates Table
# MAGIC 
# MAGIC Finally, it is very likely we will want to derive features from both the transaction log and the user activity data where we examine days without activity.  To make this analysis easier, it may be helpful to generate a table containing one record for each date from the beginning date to the end date in our dataset.  We know that these data span January 1, 2015 through March 31, 2017.  With that in mind, we can generate such a table as follows:

# COMMAND ----------

# calculate days in range
start_date = date(2015, 1, 1)
end_date = date(2017, 3, 31)
days = end_date - start_date

# generate temp view of dates in range
( spark
    .range(0, days.days)  
    .withColumn('start_date', lit(start_date.strftime('%Y-%m-%d')))  # first date in activity dataset
    .selectExpr('date_add(start_date, CAST(id as int)) as date')
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true") 
    .saveAsTable('dates')
  )

# display SQL table content
display(spark.table('dates').orderBy('date'))

# COMMAND ----------


