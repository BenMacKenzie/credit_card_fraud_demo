# Databricks notebook source
# MAGIC %sql
# MAGIC use benmackenzie_catalog.credit_card_fraud_demo

# COMMAND ----------

# MAGIC %md
# MAGIC ###We are going to build a fraud model for credit card transactions

# COMMAND ----------

# MAGIC %md
# MAGIC ###Let's start by looking out our data.  The first table is an scd type 2 dimension table

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from customers

# COMMAND ----------

# MAGIC %md
# MAGIC ###The next table is a type 1 dim table

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from merchants

# COMMAND ----------

# MAGIC %md
# MAGIC ###The transaction table along with fraud label

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from transactions

# COMMAND ----------

# MAGIC %md
# MAGIC ###Some 7 and 14 day aggregates of transaction volumes

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from transaction_aggregates

# COMMAND ----------

# MAGIC %md
# MAGIC ###We want to calculate the distance between the merchant and the customers home address.  We cannot model this as a feature because we don't know the merchant until inference time.  So we model it as a function.

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
# MAGIC ###Features:
# MAGIC 1. transaction amount
# MAGIC 2. aggregates related to spend over time windows
# MAGIC 3. distance

# COMMAND ----------

# MAGIC %md
# MAGIC ###Let define our 'EOL' table 
# MAGIC 1. pk for the entity: transaction_id
# MAGIC 2. observation date: transaction_date
# MAGIC 3. fk for customer: primary account number
# MAGIC 4. fk for merchant: merchant_id
# MAGIC 5. transaction amount
# MAGIC
# MAGIC only include 'card present' transactions

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace view fraud_eol as select transaction_id, cast(primary_account_number as bigint) as primary_account_number, cast(merchant_id as bigint) as merchant_id, transaction_date as observation_date, fraudulent as label, amount from transactions where card_present

# COMMAND ----------

from databricks.feature_engineering.entities.feature_lookup import FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction
fs = FeatureEngineeringClient()

# COMMAND ----------

feature_lookups = [
    FeatureLookup(
        table_name="benmackenzie_catalog.credit_card_fraud_demo.transaction_aggregates",
        feature_names=["14_day_total", "7_day_total", "7_day_transaction_count"],
        lookup_key="primary_account_number",
        timestamp_lookup_key = "observation_date"
    ),
    FeatureLookup(
        table_name="benmackenzie_catalog.credit_card_fraud_demo.merchants",
        feature_names=["merchant_zip"],        
        lookup_key="merchant_id"
    ),
  
   FeatureLookup(
        table_name="benmackenzie_catalog.credit_card_fraud_demo.customers",
        feature_names=["zip_code"],        
        lookup_key="primary_account_number",
        timestamp_lookup_key = "observation_date"
    ),
   
   FeatureFunction(
        udf_name="benmackenzie_catalog.credit_card_fraud_demo.distance",    
        input_bindings={
            "from_zip": "zip_code",
            "to_zip": "merchant_zip"
        },
        output_name="merchant_distance_from_home")
      
]



# COMMAND ----------

eol_df = spark.sql('select * from fraud_eol')

training_set = fs.create_training_set(
    df=eol_df,
    feature_lookups=feature_lookups,
    label = 'label',
    exclude_columns = ['transaction_id', 'customer_id', 'observation_date', 'merchant_id',  'merchant_zip', 'zip_code']
)
training_df = training_set.load_df()

# COMMAND ----------

display(training_df)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from pyspark.sql.functions import col

mlflow.set_registry_uri("databricks-uc")

transformers = []


skdtc_classifier = DecisionTreeClassifier(
  criterion="gini",
  max_depth=10
)

model = Pipeline([
    ("classifier", skdtc_classifier),
])

with mlflow.start_run():
  training_df = training_set.load_df().toPandas()
  X_train = training_df.drop(['label'], axis=1)
  y_train = training_df['label'] = training_df['label'].astype(float)
  model.fit(X_train, y_train)


fs.log_model(
    model=model,
    artifact_path="transaction_fraud",
    flavor=mlflow.sklearn,
    training_set=training_set,
    registered_model_name="transaction_fraud"
  )


# COMMAND ----------

inference_df = spark.sql('select primary_account_number, merchant_id, date("2024-04-15") as observation_date, amount from fraud_eol where observation_date = "2024-04-15"')
display(inference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Inference 

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("benmackenzie_catalog.credit_card_fraud_demo.transaction_fraud", "champion", 1)

# COMMAND ----------

#note that the URI is made up of current catalog and schema and the name of the model from fs.log_model above. 

model_version_uri = "models:/benmackenzie_catalog.credit_card_fraud_demo.transaction_fraud@champion" 
predictions = fs.score_batch(
    model_uri=model_version_uri,
    df=inference_df
)
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Online Inference
# MAGIC
# MAGIC create online tables through UI or following api. see https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#create-an-online-table-using-apis
# MAGIC
# MAGIC
# MAGIC Use databricks online store or Cosmos DB: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/publish-features#cosmosdb-compatibility
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Deploy model to serving endpoint through UI or code (reference?)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call model

# COMMAND ----------


import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/credit_card_fraud/invocations'
    headers = {'Authorization': f'Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()
    

data = [
    {
        "primary_account_number": 10048,
        "observation_date": "2024-04-15",
        "merchant_id": 20018,
        "amount": 465.25
    }
]

# Converting the list to a pandas DataFrame
df = pd.DataFrame(data)

result = score_model(df)
print(result)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ai_query('credit_card_fraud',
# MAGIC     request =>  named_struct(
# MAGIC         'primary_account_number', 10048, 
# MAGIC         'observation_date', '2024-04-15',
# MAGIC         'merchant_id', 20018, 
# MAGIC         'amount', 465
# MAGIC     ),
# MAGIC     returnType => 'Float'
# MAGIC ) AS churn

# COMMAND ----------

# MAGIC %md
# MAGIC note that I needed to use 465 instead of 465.12

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Serving

# COMMAND ----------

fs.create_feature_spec(
  name="benmackenzie_catalog.credit_card_fraud_demo.credit_fraud_features",
  features=feature_lookups,
)

# COMMAND ----------

from databricks.feature_engineering.entities.feature_serving_endpoint import (
  ServedEntity,
  EndpointCoreConfig,
)

fs.create_feature_serving_endpoint(
  name="credit-fraud-features",
    config=EndpointCoreConfig(
    served_entities=ServedEntity(
      feature_spec_name="benmackenzie_catalog.credit_card_fraud_demo.credit_fraud_features",
             workload_size="Small",
             scale_to_zero_enabled=True,
             instance_profile_arn=None,
    )
  )
)

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
response = client.predict(
    endpoint="credit-fraud-features",
    inputs={
        "dataframe_records": [
            {"primary_account_number": 10048, "observation_date": '2024-04-15', 'merchant_id': 20018, "amount": 465},
   
        ]
    },
)

print(response)