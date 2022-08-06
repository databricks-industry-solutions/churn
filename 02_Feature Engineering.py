# Databricks notebook source
# MAGIC %md The purpose of this notebook is to engineer a limited number of features from the transaction and user logs which will be used to predict churn.  

# COMMAND ----------

# MAGIC %md ###Step 1: Engineer Features from the Transaction Log
# MAGIC 
# MAGIC It is important when we access the data in the transaction log that we limit our results to information we would have access to just prior to the start of the period of interest but not after.  For example, if we are examining churn in February 2017, we would want to examine transaction log data up to and through January 31, 2017 but not on or after February 1, 2017. 
# MAGIC 
# MAGIC Knowing which subscriptions are viable headed into the period of interest and when those subscriptions started, we can define a range of dates from the start of the subscription through the day prior to the start of the period of interest from which we might derive transaction log features.  These ranges are calculated in the next cell, presented here in isolation so that the logic may more easily be reviewed before it is applied in the feature engineering query below it:

# COMMAND ----------

# DBTITLE 1,Set database for all the following queries
# please use a personalized database name here, if you wish to avoid interfering with other users who might be running this accelerator in the same workspace
database_name = 'kkbox_churn'
spark.sql(f'USE {database_name}')

# COMMAND ----------

# DBTITLE 1,Transaction Log Windows for Training Period (Feb 2017)
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   a.msno,
# MAGIC   b.subscription_id,
# MAGIC   b.subscription_start as start_at,
# MAGIC   c.last_at
# MAGIC FROM train a  -- LIMIT ANALYSIS TO AT-RISK SUBSCRIBERS IN THE TRAINING PERIOD
# MAGIC LEFT OUTER JOIN (   -- subscriptions not yet churned heading into the period of interest
# MAGIC   SELECT *
# MAGIC   FROM subscription_windows 
# MAGIC   WHERE subscription_start < cast('2017-02-01' as date)  AND subscription_end > cast('2017-02-01' as date) - interval 30 days
# MAGIC   )b
# MAGIC   ON a.msno=b.msno
# MAGIC LEFT OUTER JOIN (
# MAGIC   SELECT            -- last transaction date prior to the start of the at-risk period (we could also have just set this to the day prior to the start of the period of interest)
# MAGIC     subscription_id,
# MAGIC     MAX(transaction_date) as last_at
# MAGIC   FROM transactions_enhanced
# MAGIC   WHERE transaction_date < cast('2017-02-01' as date) 
# MAGIC   GROUP BY subscription_id
# MAGIC   ) c
# MAGIC   ON b.subscription_id=c.subscription_id

# COMMAND ----------

# MAGIC %md Using these date ranges, we can now derive features from the transaction log for the *current* subscription.  Please note, it may also be interesting to derive information from all of a subscriber's prior subscriptions, but for this exercise, we are limiting our feature engineering to information associated with the current subscription plus a simple count of prior subscriptions:

# COMMAND ----------

# DBTITLE 1,Transaction Log Features for Training Period (Feb 2017)
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS train_trans_features;
# MAGIC 
# MAGIC CREATE TABLE train_trans_features
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   WITH transaction_window (  -- this is the query from above defined as a CTE
# MAGIC     SELECT
# MAGIC       a.msno,
# MAGIC       b.subscription_id,
# MAGIC       b.subscription_start as start_at,
# MAGIC       c.last_at
# MAGIC     FROM train a
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT *
# MAGIC       FROM subscription_windows 
# MAGIC       WHERE subscription_start < cast('2017-02-01' as date)  AND subscription_end > cast('2017-02-01' as date) - interval 30 days
# MAGIC       )b
# MAGIC       ON a.msno=b.msno
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT  
# MAGIC         subscription_id,
# MAGIC         MAX(transaction_date) as last_at
# MAGIC       FROM transactions_enhanced
# MAGIC       WHERE transaction_date < cast('2017-02-01' as date) 
# MAGIC       GROUP BY subscription_id
# MAGIC       ) c
# MAGIC       ON b.subscription_id=c.subscription_id
# MAGIC       )
# MAGIC   SELECT
# MAGIC     a.msno,
# MAGIC     YEAR(b.start_at) as start_year,
# MAGIC     MONTH(b.start_at) as start_month,
# MAGIC     DATEDIFF(b.last_at, b.start_at) as subscription_age,
# MAGIC     c.renewals,
# MAGIC     c.total_list_price,
# MAGIC     c.total_amount_paid,
# MAGIC     c.total_discount,
# MAGIC     DATEDIFF(cast('2017-02-01' as date) , LAST(a.transaction_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date)) as days_since_last_account_action,
# MAGIC     LAST(a.plan_list_price) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_plan_list_price,
# MAGIC     LAST(a.actual_amount_paid) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_actual_amount_paid,
# MAGIC     LAST(a.discount) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_discount,
# MAGIC     LAST(a.payment_plan_days) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_payment_plan_days,
# MAGIC     LAST(a.payment_method_id) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_payment_method,
# MAGIC     LAST(a.is_cancel) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_is_cancel,
# MAGIC     LAST(a.is_auto_renew) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_is_auto_renew,
# MAGIC     LAST(a.change_in_list_price) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_list_price,
# MAGIC     LAST(a.change_in_discount) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_discount,
# MAGIC     LAST(a.change_in_payment_plan_days) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_payment_plan_days,
# MAGIC     LAST(a.change_in_payment_method_id) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_payment_method_id,
# MAGIC     LAST(a.change_in_cancellation) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_cancellation,
# MAGIC     LAST(a.change_in_auto_renew) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_auto_renew,
# MAGIC     LAST(a.days_change_in_membership_expire_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_days_change_in_membership_expire_date,
# MAGIC     DATEDIFF(cast('2017-02-01' as date) , LAST(a.membership_expire_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date)) as days_until_expiration,
# MAGIC     d.total_subscription_count,
# MAGIC     e.city,
# MAGIC     CASE WHEN e.bd < 10 THEN NULL WHEN e.bd > 70 THEN NULL ELSE e.bd END as bd,
# MAGIC     CASE WHEN LOWER(e.gender)='female' THEN 0 WHEN LOWER(e.gender)='male' THEN 1 ELSE NULL END as gender,
# MAGIC     e.registered_via  
# MAGIC   FROM transactions_enhanced a
# MAGIC   INNER JOIN transaction_window b
# MAGIC     ON a.subscription_id=b.subscription_id AND a.transaction_date = b.last_at
# MAGIC   INNER JOIN (
# MAGIC     SELECT  -- summary stats for current subscription
# MAGIC       x.subscription_id,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.plan_list_price ELSE 0 END) as total_list_price,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.actual_amount_paid ELSE 0 END) as total_amount_paid,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.discount ELSE 0 END) as total_discount,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN 1 ELSE 0 END) as renewals
# MAGIC     FROM transactions_enhanced x
# MAGIC     INNER JOIN transaction_window y
# MAGIC       ON x.subscription_id=y.subscription_id AND x.transaction_date BETWEEN y.start_at AND y.last_at
# MAGIC     GROUP BY x.subscription_id
# MAGIC     ) c
# MAGIC     ON a.subscription_id=c.subscription_id
# MAGIC   INNER JOIN (
# MAGIC     SELECT  -- count of all unique subscriptions for each customer
# MAGIC       msno,
# MAGIC       COUNT(*) as total_subscription_count
# MAGIC     FROM subscription_windows
# MAGIC     WHERE subscription_start < cast('2017-02-01' as date) 
# MAGIC     GROUP BY msno
# MAGIC     ) d
# MAGIC     ON a.msno=d.msno
# MAGIC   LEFT OUTER JOIN members e
# MAGIC     ON a.msno=e.msno;
# MAGIC     
# MAGIC SELECT * FROM train_trans_features;

# COMMAND ----------

# MAGIC %md Modifying the dates, we can derive these same features for the test period, March 2017:

# COMMAND ----------

# DBTITLE 1,Transaction Log Features for Testing Period (Mar 2017)
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS test_trans_features;
# MAGIC 
# MAGIC CREATE TABLE test_trans_features
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   WITH transaction_window (
# MAGIC     SELECT
# MAGIC       a.msno,
# MAGIC       b.subscription_id,
# MAGIC       b.subscription_start as start_at,
# MAGIC       c.last_at
# MAGIC     FROM test a  -- LIMIT ANALYSIS TO AT-RISK SUBSCRIBERS IN THE TESTING PERIOD
# MAGIC     LEFT OUTER JOIN (  -- subscriptions not yet churned heading into the period of interest
# MAGIC       SELECT *
# MAGIC       FROM subscription_windows 
# MAGIC       WHERE subscription_start < '2017-03-01' AND subscription_end > cast('2017-03-01' as date) - interval 30 days
# MAGIC       )b
# MAGIC       ON a.msno=b.msno
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT  
# MAGIC         subscription_id,
# MAGIC         MAX(transaction_date) as last_at
# MAGIC       FROM transactions_enhanced
# MAGIC       WHERE transaction_date < '2017-03-01'
# MAGIC       GROUP BY subscription_id
# MAGIC       ) c
# MAGIC       ON b.subscription_id=c.subscription_id
# MAGIC       )
# MAGIC   SELECT
# MAGIC     a.msno,
# MAGIC     YEAR(b.start_at) as start_year,
# MAGIC     MONTH(b.start_at) as start_month,
# MAGIC     DATEDIFF(b.last_at, b.start_at) as subscription_age,
# MAGIC     c.renewals,
# MAGIC     c.total_list_price,
# MAGIC     c.total_amount_paid,
# MAGIC     c.total_discount,
# MAGIC     DATEDIFF('2017-03-01', LAST(a.transaction_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date)) as days_since_last_account_action,
# MAGIC     LAST(a.plan_list_price) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_plan_list_price,
# MAGIC     LAST(a.actual_amount_paid) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_actual_amount_paid,
# MAGIC     LAST(a.discount) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_discount,
# MAGIC     LAST(a.payment_plan_days) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_payment_plan_days,
# MAGIC     LAST(a.payment_method_id) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_payment_method,
# MAGIC     LAST(a.is_cancel) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_is_cancel,
# MAGIC     LAST(a.is_auto_renew) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_is_auto_renew,
# MAGIC     LAST(a.change_in_list_price) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_list_price,
# MAGIC     LAST(a.change_in_discount) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_discount,
# MAGIC     LAST(a.change_in_payment_plan_days) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_payment_plan_days,
# MAGIC     LAST(a.change_in_payment_method_id) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_payment_method_id,
# MAGIC     LAST(a.change_in_cancellation) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_cancellation,
# MAGIC     LAST(a.change_in_auto_renew) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_auto_renew,
# MAGIC     LAST(a.days_change_in_membership_expire_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_days_change_in_membership_expire_date,
# MAGIC     DATEDIFF('2017-03-01', LAST(a.membership_expire_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date)) as days_until_expiration,
# MAGIC     d.total_subscription_count,
# MAGIC     e.city,
# MAGIC     CASE WHEN e.bd < 10 THEN NULL WHEN e.bd > 70 THEN NULL ELSE e.bd END as bd,
# MAGIC     CASE WHEN LOWER(e.gender)='female' THEN 0 WHEN LOWER(e.gender)='male' THEN 1 ELSE NULL END as gender,
# MAGIC     e.registered_via  
# MAGIC   FROM transactions_enhanced a
# MAGIC   INNER JOIN transaction_window b
# MAGIC     ON a.subscription_id=b.subscription_id AND a.transaction_date = b.last_at
# MAGIC   INNER JOIN (
# MAGIC     SELECT 
# MAGIC       x.subscription_id,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.plan_list_price ELSE 0 END) as total_list_price,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.actual_amount_paid ELSE 0 END) as total_amount_paid,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.discount ELSE 0 END) as total_discount,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN 1 ELSE 0 END) as renewals
# MAGIC     FROM transactions_enhanced x
# MAGIC     INNER JOIN transaction_window y
# MAGIC       ON x.subscription_id=y.subscription_id AND x.transaction_date BETWEEN y.start_at AND y.last_at
# MAGIC     GROUP BY x.subscription_id
# MAGIC     ) c
# MAGIC     ON a.subscription_id=c.subscription_id
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       msno,
# MAGIC       COUNT(*) as total_subscription_count
# MAGIC     FROM subscription_windows
# MAGIC     WHERE subscription_start < '2017-03-01'
# MAGIC     GROUP BY msno
# MAGIC     ) d
# MAGIC     ON a.msno=d.msno
# MAGIC   LEFT OUTER JOIN members e
# MAGIC     ON a.msno=e.msno;
# MAGIC     
# MAGIC SELECT * FROM test_trans_features;

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Examining the transaction features above, you may recognize the opportunity to derive many more features.  Our goal in this exercise is not to provide an exhaustive review of feature types but instead to generate a meaningful subset of potential features against which to train our model.
# MAGIC 
# MAGIC Before going further, let's make sure we have features for all customers identified in our training and testing period datasets.  Each of these queries should return a count of zero unmatched records:

# COMMAND ----------

# DBTITLE 1,Features for All Training Subscribers
# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*)
# MAGIC FROM train a
# MAGIC LEFT OUTER JOIN train_trans_features b
# MAGIC   ON a.msno=b.msno
# MAGIC WHERE b.msno IS NULL

# COMMAND ----------

# DBTITLE 1,Features for All Testing Subscribers
# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*)
# MAGIC FROM test a
# MAGIC LEFT OUTER JOIN test_trans_features b
# MAGIC   ON a.msno=b.msno
# MAGIC WHERE b.msno IS NULL

# COMMAND ----------

# MAGIC %md ###Step 2: Engineer Features from the User Logs
# MAGIC 
# MAGIC As with our transaction log features, we need to define a range of dates within which we wish to examine user activity for the current, at-risk subscription.  This logic differs from the earlier logic in that we'll consider all user activity headed into the period of interest as KKBox allows users to continue using their subscription for 30-days following expiration.  Knowing an expired subscription is still in use should be a significant indication of churn intent.  
# MAGIC 
# MAGIC In addition, it should be noted that we are constraining our feature generation from the user logs to activity occurring no more than 30-days prior to the start of the period of interest. As with the transaction logs, there are many more features we could derive such as those representing usage at the beginning of the subscription, usage throughout the subscription (ahead of the start of the period of interest), and periods of differing durations heading into the period of interest.  The limiting of the features in this way is arbitrary as, again, our goal in this exercise is not to create an exhaustive set of features but to create a meaningful set which could be used in model training. 
# MAGIC 
# MAGIC With all that in mind, let's calculate the date ranges over which we will derive features from the user logs:

# COMMAND ----------

# DBTITLE 1,User Log Windows for Training Period (Feb 2017)
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   a.msno,
# MAGIC   b.subscription_id,
# MAGIC   CASE 
# MAGIC     WHEN b.subscription_start < cast('2017-02-01' as date) - interval 30 days THEN cast('2017-02-01' as date) - interval 30 days-- cap subscription info to 30-days prior to start of period
# MAGIC     ELSE b.subscription_start 
# MAGIC     END as start_at,
# MAGIC   cast('2017-02-01' as date) - interval 1 day as end_at,
# MAGIC   c.last_at as last_exp_at
# MAGIC FROM train a  -- LIMIT ANALYSIS TO AT-RISK SUBSCRIBERS IN THE TRAINING PERIOD
# MAGIC LEFT OUTER JOIN (   -- subscriptions not yet churned heading into the period of interest 
# MAGIC   SELECT *
# MAGIC   FROM subscription_windows 
# MAGIC   WHERE subscription_start < cast('2017-02-01' as date)  AND subscription_end > cast('2017-02-01' as date) - interval 30 days
# MAGIC   )b
# MAGIC   ON a.msno=b.msno
# MAGIC LEFT OUTER JOIN (  -- last known expiration date headed into this period
# MAGIC   SELECT
# MAGIC     x.subscription_id,
# MAGIC     y.membership_expire_date as last_at
# MAGIC   FROM (
# MAGIC     SELECT  -- last subscription transaction before start of this period
# MAGIC       subscription_id,
# MAGIC       MAX(transaction_date) as transaction_date
# MAGIC     FROM transactions_enhanced
# MAGIC     WHERE transaction_date < cast('2017-02-01' as date) 
# MAGIC     GROUP BY subscription_id
# MAGIC     ) x
# MAGIC   INNER JOIN transactions_enhanced y
# MAGIC     ON x.subscription_id=y.subscription_id AND x.transaction_date=y.transaction_date
# MAGIC   ) c
# MAGIC   ON b.subscription_id=c.subscription_id  

# COMMAND ----------

# MAGIC %md Using these date ranges, we can now constrain our analysis of the user logs.  It's important to note that users may have multiple streaming sessions on a given date.  As such, we'll want to derive day-level statistics on the user-logs to make them easier to consume.  In addition, we will want to join our day-level statistics with our date range dataset, *i.e.* dates derived in the last notebook, so that we may have one record for each day in the range of interest.  Understanding patterns of activity as well as inactivity may be helpful in determining which subscriptions will churn: 

# COMMAND ----------

# DBTITLE 1,Calculate Day-Level User Activity Stats (Feb 2017)
# MAGIC %sql
# MAGIC 
# MAGIC WITH activity_window (
# MAGIC     SELECT
# MAGIC       a.msno,
# MAGIC       b.subscription_id,
# MAGIC       CASE 
# MAGIC         WHEN b.subscription_start < cast('2017-02-01' as date)  - interval 30 days THEN cast('2017-02-01' as date)  - interval 30 days 
# MAGIC         ELSE b.subscription_start 
# MAGIC         END as start_at,
# MAGIC       cast('2017-02-01' as date)  - interval 1 day as end_at,
# MAGIC       c.last_at as last_exp_at
# MAGIC     FROM train a
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT *
# MAGIC       FROM subscription_windows 
# MAGIC       WHERE subscription_start < cast('2017-02-01' as date)  AND subscription_end > cast('2017-02-01' as date)  - interval 30 days
# MAGIC       )b
# MAGIC       ON a.msno=b.msno
# MAGIC     LEFT OUTER JOIN (  -- last known expiration date headed into this period
# MAGIC       SELECT
# MAGIC         x.subscription_id,
# MAGIC         y.membership_expire_date as last_at
# MAGIC       FROM (
# MAGIC         SELECT  -- last subscription transaction before start of this period
# MAGIC           subscription_id,
# MAGIC           MAX(transaction_date) as transaction_date
# MAGIC         FROM transactions_enhanced
# MAGIC         WHERE transaction_date < cast('2017-02-01' as date) 
# MAGIC         GROUP BY subscription_id
# MAGIC         ) x
# MAGIC       INNER JOIN transactions_enhanced y
# MAGIC         ON x.subscription_id=y.subscription_id AND x.transaction_date=y.transaction_date
# MAGIC       ) c
# MAGIC       ON b.subscription_id=c.subscription_id  
# MAGIC     )    
# MAGIC SELECT
# MAGIC   a.subscription_id,
# MAGIC   a.msno,
# MAGIC   b.date,
# MAGIC   CASE WHEN b.date > a.last_exp_at THEN 1 ELSE 0 END as after_exp,
# MAGIC   CASE WHEN c.date IS NOT NULL THEN 1 ELSE 0 END as had_session,
# MAGIC   COALESCE(c.session_count, 0) as sessions_total,
# MAGIC   COALESCE(c.total_secs, 0) as seconds_total,
# MAGIC   COALESCE(c.num_uniq,0) as number_uniq,
# MAGIC   COALESCE(c.num_total,0) as number_total
# MAGIC FROM activity_window a
# MAGIC INNER JOIN dates b
# MAGIC   ON b.date BETWEEN a.start_at AND a.end_at
# MAGIC LEFT OUTER JOIN (
# MAGIC   SELECT
# MAGIC     msno,
# MAGIC     date,
# MAGIC     COUNT(*) as session_count,
# MAGIC     SUM(total_secs) as total_secs,
# MAGIC     SUM(num_uniq) as num_uniq,
# MAGIC     SUM(num_25+num_50+num_75+num_985+num_100) as num_total
# MAGIC   FROM user_logs
# MAGIC   GROUP BY msno, date
# MAGIC   ) c
# MAGIC   ON a.msno=c.msno AND b.date=c.date
# MAGIC ORDER BY subscription_id, date

# COMMAND ----------

# MAGIC %md With our daily activity records now constructed, we can create the summary statistics that will form our user activity features:

# COMMAND ----------

# DBTITLE 1,User Activity Log Features for Training Period (Feb 2017)
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS train_act_features;
# MAGIC 
# MAGIC CREATE TABLE train_act_features
# MAGIC USING DELTA 
# MAGIC AS
# MAGIC WITH activity_window (
# MAGIC     SELECT
# MAGIC       a.msno,
# MAGIC       b.subscription_id,
# MAGIC       CASE 
# MAGIC         WHEN b.subscription_start < cast('2017-02-01' as date)  - interval 30 days THEN cast('2017-02-01' as date)  - interval 30 days 
# MAGIC         ELSE b.subscription_start 
# MAGIC         END as start_at,
# MAGIC       cast('2017-02-01' as date)  - interval 1 day as end_at,
# MAGIC       c.last_at as last_exp_at
# MAGIC     FROM train a
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT *
# MAGIC       FROM subscription_windows 
# MAGIC       WHERE subscription_start < cast('2017-02-01' as date)  AND subscription_end > cast('2017-02-01' as date)  - interval 30 days
# MAGIC       )b
# MAGIC       ON a.msno=b.msno
# MAGIC     LEFT OUTER JOIN (  -- last known expiration date headed into this period
# MAGIC       SELECT
# MAGIC         x.subscription_id,
# MAGIC         y.membership_expire_date as last_at
# MAGIC       FROM (
# MAGIC         SELECT  -- last subscription transaction before start of this period
# MAGIC           subscription_id,
# MAGIC           MAX(transaction_date) as transaction_date
# MAGIC         FROM transactions_enhanced
# MAGIC         WHERE transaction_date < cast('2017-02-01' as date) 
# MAGIC         GROUP BY subscription_id
# MAGIC         ) x
# MAGIC       INNER JOIN transactions_enhanced y
# MAGIC         ON x.subscription_id=y.subscription_id AND x.transaction_date=y.transaction_date
# MAGIC       ) c
# MAGIC       ON b.subscription_id=c.subscription_id  
# MAGIC       ),
# MAGIC   activity (
# MAGIC     SELECT
# MAGIC       a.subscription_id,
# MAGIC       a.msno,
# MAGIC       b.date,
# MAGIC       CASE WHEN b.date > a.last_exp_at THEN 1 ELSE 0 END as after_exp,
# MAGIC       CASE WHEN c.date IS NOT NULL THEN 1 ELSE 0 END as had_session,
# MAGIC       COALESCE(c.session_count, 0) as sessions_total,
# MAGIC       COALESCE(c.total_secs, 0) as seconds_total,
# MAGIC       COALESCE(c.num_uniq,0) as number_uniq,
# MAGIC       COALESCE(c.num_total,0) as number_total
# MAGIC     FROM activity_window a
# MAGIC     INNER JOIN dates b
# MAGIC       ON b.date BETWEEN a.start_at AND a.end_at
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         date,
# MAGIC         COUNT(*) as session_count,
# MAGIC         SUM(total_secs) as total_secs,
# MAGIC         SUM(num_uniq) as num_uniq,
# MAGIC         SUM(num_25+num_50+num_75+num_985+num_100) as num_total
# MAGIC       FROM user_logs
# MAGIC       GROUP BY msno, date
# MAGIC       ) c
# MAGIC       ON a.msno=c.msno AND b.date=c.date
# MAGIC     )
# MAGIC   
# MAGIC SELECT 
# MAGIC   subscription_id,
# MAGIC   msno,
# MAGIC   COUNT(*) as days_total,
# MAGIC   SUM(had_session) as days_with_session,
# MAGIC   COALESCE(SUM(had_session)/COUNT(*),0) as ratio_days_with_session_to_days,
# MAGIC   SUM(after_exp) as days_after_exp,
# MAGIC   SUM(had_session * after_exp) as days_after_exp_with_session,
# MAGIC   COALESCE(SUM(had_session * after_exp)/SUM(after_exp),0) as ratio_days_after_exp_with_session_to_days_after_exp,
# MAGIC   SUM(sessions_total) as sessions_total,
# MAGIC   COALESCE(SUM(sessions_total)/COUNT(*),0) as ratio_sessions_total_to_days_total,
# MAGIC   COALESCE(SUM(sessions_total)/SUM(had_session),0) as ratio_sessions_total_to_days_with_session,
# MAGIC   SUM(sessions_total * after_exp) as sessions_total_after_exp,
# MAGIC   COALESCE(SUM(sessions_total * after_exp)/SUM(after_exp),0) as ratio_sessions_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(sessions_total * after_exp)/SUM(had_session * after_exp),0) as ratio_sessions_total_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(seconds_total) as seconds_total,
# MAGIC   COALESCE(SUM(seconds_total)/COUNT(*),0) as ratio_seconds_total_to_days_total,
# MAGIC   COALESCE(SUM(seconds_total)/SUM(had_session),0) as ratio_seconds_total_to_days_with_session,
# MAGIC   SUM(seconds_total * after_exp) as seconds_total_after_exp,
# MAGIC   COALESCE(SUM(seconds_total * after_exp)/SUM(after_exp),0) as ratio_seconds_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(seconds_total * after_exp)/SUM(had_session * after_exp),0) as ratio_seconds_total_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(number_uniq) as number_uniq,
# MAGIC   COALESCE(SUM(number_uniq)/COUNT(*),0) as ratio_number_uniq_to_days_total,
# MAGIC   COALESCE(SUM(number_uniq)/SUM(had_session),0) as ratio_number_uniq_to_days_with_session,
# MAGIC   SUM(number_uniq * after_exp) as number_uniq_after_exp,
# MAGIC   COALESCE(SUM(number_uniq * after_exp)/SUM(after_exp),0) as ratio_number_uniq_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(number_uniq * after_exp)/SUM(had_session * after_exp),0) as ratio_number_uniq_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(number_total) as number_total,
# MAGIC   COALESCE(SUM(number_total)/COUNT(*),0) as ratio_number_total_to_days_total,
# MAGIC   COALESCE(SUM(number_total)/SUM(had_session),0) as ratio_number_total_to_days_with_session,
# MAGIC   SUM(number_total * after_exp) as number_total_after_exp,
# MAGIC   COALESCE(SUM(number_total * after_exp)/SUM(after_exp),0) as ratio_number_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(number_total * after_exp)/SUM(had_session * after_exp),0) as ratio_number_total_after_exp_to_days_after_exp_with_session
# MAGIC FROM activity
# MAGIC GROUP BY subscription_id, msno
# MAGIC ORDER BY msno;
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM train_act_features;

# COMMAND ----------

# MAGIC %md We can use the same logic to generate features for the testing period:

# COMMAND ----------

# DBTITLE 1,User Activity Log Features for Testing Period (Mar 2017)
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS test_act_features;
# MAGIC 
# MAGIC CREATE TABLE test_act_features
# MAGIC USING DELTA 
# MAGIC AS
# MAGIC WITH activity_window (
# MAGIC     SELECT
# MAGIC       a.msno,
# MAGIC       b.subscription_id,
# MAGIC       CASE 
# MAGIC         WHEN b.subscription_start <'2017-03-01' - interval 30 days THEN'2017-03-01' - interval 30 days 
# MAGIC         ELSE b.subscription_start 
# MAGIC         END as start_at,
# MAGIC      '2017-03-01' - interval 1 day as end_at,
# MAGIC       c.last_at as last_exp_at
# MAGIC     FROM test a
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT *
# MAGIC       FROM subscription_windows 
# MAGIC       WHERE subscription_start < '2017-03-01' AND subscription_end >'2017-03-01' - interval 30 days
# MAGIC       )b
# MAGIC       ON a.msno=b.msno
# MAGIC     LEFT OUTER JOIN (  -- last known expiration date headed into this period
# MAGIC       SELECT
# MAGIC         x.subscription_id,
# MAGIC         y.membership_expire_date as last_at
# MAGIC       FROM (
# MAGIC         SELECT  -- last subscription transaction before start of this period
# MAGIC           subscription_id,
# MAGIC           MAX(transaction_date) as transaction_date
# MAGIC         FROM transactions_enhanced
# MAGIC         WHERE transaction_date < '2017-03-01'
# MAGIC         GROUP BY subscription_id
# MAGIC         ) x
# MAGIC       INNER JOIN transactions_enhanced y
# MAGIC         ON x.subscription_id=y.subscription_id AND x.transaction_date=y.transaction_date
# MAGIC       ) c
# MAGIC       ON b.subscription_id=c.subscription_id  
# MAGIC       ),
# MAGIC   activity (
# MAGIC     SELECT
# MAGIC       a.subscription_id,
# MAGIC       a.msno,
# MAGIC       b.date,
# MAGIC       CASE WHEN b.date > a.last_exp_at THEN 1 ELSE 0 END as after_exp,
# MAGIC       CASE WHEN c.date IS NOT NULL THEN 1 ELSE 0 END as had_session,
# MAGIC       COALESCE(c.session_count, 0) as sessions_total,
# MAGIC       COALESCE(c.total_secs, 0) as seconds_total,
# MAGIC       COALESCE(c.num_uniq,0) as number_uniq,
# MAGIC       COALESCE(c.num_total,0) as number_total
# MAGIC     FROM activity_window a
# MAGIC     INNER JOIN dates b
# MAGIC       ON b.date BETWEEN a.start_at AND a.end_at
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         date,
# MAGIC         COUNT(*) as session_count,
# MAGIC         SUM(total_secs) as total_secs,
# MAGIC         SUM(num_uniq) as num_uniq,
# MAGIC         SUM(num_25+num_50+num_75+num_985+num_100) as num_total
# MAGIC       FROM user_logs
# MAGIC       GROUP BY msno, date
# MAGIC       ) c
# MAGIC       ON a.msno=c.msno AND b.date=c.date
# MAGIC     )
# MAGIC   
# MAGIC SELECT 
# MAGIC   subscription_id,
# MAGIC   msno,
# MAGIC   COUNT(*) as days_total,
# MAGIC   SUM(had_session) as days_with_session,
# MAGIC   COALESCE(SUM(had_session)/COUNT(*),0) as ratio_days_with_session_to_days,
# MAGIC   SUM(after_exp) as days_after_exp,
# MAGIC   SUM(had_session * after_exp) as days_after_exp_with_session,
# MAGIC   COALESCE(SUM(had_session * after_exp)/SUM(after_exp),0) as ratio_days_after_exp_with_session_to_days_after_exp,
# MAGIC   SUM(sessions_total) as sessions_total,
# MAGIC   COALESCE(SUM(sessions_total)/COUNT(*),0) as ratio_sessions_total_to_days_total,
# MAGIC   COALESCE(SUM(sessions_total)/SUM(had_session),0) as ratio_sessions_total_to_days_with_session,
# MAGIC   SUM(sessions_total * after_exp) as sessions_total_after_exp,
# MAGIC   COALESCE(SUM(sessions_total * after_exp)/SUM(after_exp),0) as ratio_sessions_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(sessions_total * after_exp)/SUM(had_session * after_exp),0) as ratio_sessions_total_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(seconds_total) as seconds_total,
# MAGIC   COALESCE(SUM(seconds_total)/COUNT(*),0) as ratio_seconds_total_to_days_total,
# MAGIC   COALESCE(SUM(seconds_total)/SUM(had_session),0) as ratio_seconds_total_to_days_with_session,
# MAGIC   SUM(seconds_total * after_exp) as seconds_total_after_exp,
# MAGIC   COALESCE(SUM(seconds_total * after_exp)/SUM(after_exp),0) as ratio_seconds_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(seconds_total * after_exp)/SUM(had_session * after_exp),0) as ratio_seconds_total_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(number_uniq) as number_uniq,
# MAGIC   COALESCE(SUM(number_uniq)/COUNT(*),0) as ratio_number_uniq_to_days_total,
# MAGIC   COALESCE(SUM(number_uniq)/SUM(had_session),0) as ratio_number_uniq_to_days_with_session,
# MAGIC   SUM(number_uniq * after_exp) as number_uniq_after_exp,
# MAGIC   COALESCE(SUM(number_uniq * after_exp)/SUM(after_exp),0) as ratio_number_uniq_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(number_uniq * after_exp)/SUM(had_session * after_exp),0) as ratio_number_uniq_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(number_total) as number_total,
# MAGIC   COALESCE(SUM(number_total)/COUNT(*),0) as ratio_number_total_to_days_total,
# MAGIC   COALESCE(SUM(number_total)/SUM(had_session),0) as ratio_number_total_to_days_with_session,
# MAGIC   SUM(number_total * after_exp) as number_total_after_exp,
# MAGIC   COALESCE(SUM(number_total * after_exp)/SUM(after_exp),0) as ratio_number_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(number_total * after_exp)/SUM(had_session * after_exp),0) as ratio_number_total_after_exp_to_days_after_exp_with_session
# MAGIC FROM activity
# MAGIC GROUP BY subscription_id, msno
# MAGIC ORDER BY msno;
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM test_act_features;

# COMMAND ----------

# MAGIC %md And again, let's make sure we aren't missing records for any at-risk subscriptions.  Each of these queries should return a count of zero:

# COMMAND ----------

# DBTITLE 1,Features for All Training Subscribers
# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*)
# MAGIC FROM train a
# MAGIC LEFT OUTER JOIN train_act_features b
# MAGIC   ON a.msno=b.msno
# MAGIC WHERE b.msno IS NULL

# COMMAND ----------

# DBTITLE 1,Features for All Testing Subscribers
# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*)
# MAGIC FROM test a
# MAGIC LEFT OUTER JOIN test_act_features b
# MAGIC   ON a.msno=b.msno
# MAGIC WHERE b.msno IS NULL
