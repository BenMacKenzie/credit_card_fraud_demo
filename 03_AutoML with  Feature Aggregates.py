# Databricks notebook source
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install --force-reinstall databricks-feature-engineering==0.8.0a2 catboost
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = 'benmackenzie_catalog'
schema = 'credit_card_fraud_demo'

# COMMAND ----------

spark.sql(f"use {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Look at Data

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables

# COMMAND ----------

# MAGIC %sql
# MAGIC select max(transaction_date) as max, min(transaction_date) as min from transactions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from itertools import product
import datetime
from databricks.feature_engineering import FeatureAggregations, Aggregation, Window, CronSchedule
from databricks.feature_engineering.entities.aggregation_function import ApproxCountDistinct
from databricks.feature_engineering.entities.feature_lookup import FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Features Across Search Space

# COMMAND ----------



def generate_features(search_space):
  aggregations = []
  for f in search_space:
    if "offset" in f:
      combinations = product(f["window"],f["offset"], f["function"], )
     
      for row in combinations:
        window = Window(duration=datetime.timedelta(days=row[0]), offset=-datetime.timedelta(days=row[1]))
        aggregations.append(Aggregation(column=f["column"], function=row[2], window=window))
    else:
      combinations = product(f["window"], f["function"])
      for row in combinations:
        window = Window(duration=datetime.timedelta(days=row[0]))
        aggregations.append(Aggregation(column=f["column"], function=row[1], window=window))
  return aggregations


def get_feature_aggregations(source_table, lookup_key, timestamp_key, start_time, end_time, granularity, search_space):
  features = generate_features(search_space)
  aggregations =  FeatureAggregations(
    source_table=source_table,
    lookup_key=lookup_key,
    timestamp_key=timestamp_key,
    start_time=start_time,
    end_time=end_time,
    granularity=granularity,
    aggregations = features
  )
  return aggregations
  



# COMMAND ----------


search_space = [
  {"column": "amount", "window": [15, 30], "function": ["avg", "sum"]},
  {"column": "amount", "window": [5, 10], "offset": [5,10], "function": ["avg", "sum"]}
]

#note that we are creating a feature for every account_holder for every day...this is somewhat wasteful during the feature discovery phase
aggregations = get_feature_aggregations(
  source_table='benmackenzie_catalog.credit_card_fraud_demo.transactions',
  lookup_key='primary_account_number',
  timestamp_key='transaction_date',
  start_time=datetime.datetime(1974, 10, 26),
  end_time=None,
  granularity=datetime.timedelta(days=1),
  search_space=search_space)

# COMMAND ----------

aggregates_df = fe.aggregate_features(
    features=aggregations,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Define EOL Table, Regular Feature Lookups

# COMMAND ----------

# MAGIC %md
# MAGIC #### EOL table

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace view fraud_eol as select transaction_id, cast(primary_account_number as bigint) as primary_account_number, cast(merchant_id as bigint) as merchant_id, transaction_date as observation_date, fraudulent as label, amount from transactions where card_present

# COMMAND ----------

# MAGIC %md
# MAGIC #### Distance Function

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE or replace FUNCTION benmackenzie_catalog.credit_card_fraud_demo.distance(from_zip string, to_zip string)
# MAGIC RETURNS double
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'distance between two zip codes'
# MAGIC AS $$
# MAGIC import random
# MAGIC def distance(n1: str, n2: str) -> int:
# MAGIC    return random.randint(1,250)
# MAGIC
# MAGIC return distance(from_zip, to_zip)
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Lookups

# COMMAND ----------

feature_lookups = [
    FeatureLookup(
        table_name=f"{catalog}.{schema}.merchants",
        feature_names=["merchant_zip"],        
        lookup_key="merchant_id"
    ),
  
   FeatureLookup(
        table_name=f"{catalog}.{schema}.customers",
        feature_names=["zip_code"],        
        lookup_key="primary_account_number",
        timestamp_lookup_key = "observation_date"
    ),
   
   FeatureFunction(
        udf_name=f"{catalog}.{schema}.distance",    
        input_bindings={
            "from_zip": "zip_code",
            "to_zip": "merchant_zip"
        },
        output_name="merchant_distance_from_home")
      
]

# COMMAND ----------

eol_df = spark.sql('select * from fraud_eol')


# COMMAND ----------

# MAGIC %md
# MAGIC #### Merge EOL Dataframe and Aggregates Dataframe

# COMMAND ----------


aggregates_df = aggregates_df.withColumnRenamed("primary_account_number", "primary_account_number_2")

result_df = eol_df.join(
    aggregates_df,
    (eol_df["primary_account_number"] == aggregates_df["primary_account_number_2"]) &
    (eol_df["observation_date"] == aggregates_df["transaction_date"]),
    how="left"
)


result_df = result_df.drop("primary_account_number_2")




# COMMAND ----------

display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Training Data Set

# COMMAND ----------

training_set = fe.create_training_set(
    df=result_df,
    feature_lookups=feature_lookups,
    label = 'label',
    exclude_columns = ['transaction_id', 'customer_id', 'observation_date', 'transaction_date', 'merchant_id',  'merchant_zip', 'zip_code']
)
training_df = training_set.load_df()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Caboost Model and use select_features

# COMMAND ----------

from sklearn.model_selection import train_test_split
df = training_df.toPandas().dropna() #feature generation should put in defaults
target = df.pop('label')
X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=0.8)


# COMMAND ----------

from catboost import Pool

categories = []
train_pool = Pool(X_train, y_train, cat_features=categories)
eval_pool = Pool(X_test, y_test, cat_features=categories)


# COMMAND ----------

from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=300, learning_rate=0.1, random_seed=123)

selected_features = model.select_features(
                X=train_pool,
                eval_set=eval_pool,
                num_features_to_select=5,
                features_for_select = 
                  [
                  "amount_avg_15d",
                  "amount_sum_15d",
                  "amount_avg_30d",
                  "amount_sum_30d",
                  "amount_avg_5d_offset_5d",
                  "amount_sum_5d_offset_5d",
                  "amount_avg_5d_offset_10d",
                  "amount_sum_5d_offset_10d",
                  "amount_avg_10d_offset_5d",
                  "amount_sum_10d_offset_5d",
                  "amount_avg_10d_offset_10d",
                  "amount_sum_10d_offset_10d"
                  ]
                
              )

# COMMAND ----------

selected_features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC use selected features and and automl to find the best model (or just use catboost)
