# Databricks notebook source
# MAGIC %pip install --force-reinstall databricks-feature-engineering==0.8.0a2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC use bmac.credit_card_fraud_demo

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from transactions

# COMMAND ----------

import datetime
from databricks.feature_engineering import FeatureAggregations, Aggregation, Window, CronSchedule
from databricks.feature_engineering.entities.aggregation_function import ApproxCountDistinct

aggregations = FeatureAggregations(
    source_table='bmac.credit_card_fraud_demo.transactions',
    lookup_key="primary_account_number",
    timestamp_key="transaction_date",
    start_time=datetime.datetime(2023, 1, 1),
    end_time=None,
    granularity=datetime.timedelta(days=1),
    aggregations=[
        Aggregation(
            column="amount",
            function="avg", # Use shorthand string to specify the aggregation function.
            window=Window(duration=datetime.timedelta(days=30)),
        ),
        Aggregation(
            column="amount",
            function="sum",
            window=Window(duration=datetime.timedelta(days=5), offset=datetime.timedelta(days=-2))
          ),
        
    ],
)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# COMMAND ----------

df = fe.aggregate_features(
    features=aggregations,
)
display(df)
