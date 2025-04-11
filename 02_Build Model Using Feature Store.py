# Databricks notebook source
pip install databricks-feature-engineering


# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------


catalog = 'benmackenzie_catalog'
schema = 'credit_card_fraud_demo'

# COMMAND ----------

spark.sql(f"use {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###We are going to build a fraud model for credit card transactions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Customers:  scd type 2 dimension table

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from customers

# COMMAND ----------

# MAGIC %md
# MAGIC ###Merhcants: Type 1 SCD table

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
# MAGIC ### Define Function to calcuate distance between POS and customer home address
# MAGIC
# MAGIC We want to calculate the distance between the merchant and the customers home address.  We cannot model this as a feature because we don't know the merchant until inference time.  So we model it as a function.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE or replace FUNCTION benmackenzie_catalog.promote_model_demo.distance(from_zip string, to_zip string)
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
# MAGIC ## Build Training Data Set
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Define  'EOL' table 
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
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Feature Lookups

# COMMAND ----------

feature_lookups = [
    FeatureLookup(
        table_name=f"{catalog}.{schema}.transaction_aggregates",
        feature_names=["14_day_total", "7_day_total", "7_day_transaction_count"],
        lookup_key="primary_account_number",
        timestamp_lookup_key = "observation_date"
    ),
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

# MAGIC %md
# MAGIC ###Build Training Dataset

# COMMAND ----------

eol_df = spark.sql('select * from fraud_eol')

training_set = fe.create_training_set(
    df=eol_df,
    feature_lookups=feature_lookups,
    label = 'label',
    exclude_columns = ['transaction_id', 'customer_id', 'observation_date', 'merchant_id',  'merchant_zip', 'zip_code']
)
training_df = training_set.load_df()

# COMMAND ----------

training_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.training_ds")

# COMMAND ----------

display(training_df)

# COMMAND ----------

training_df = training_df.toPandas()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Model

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
from mlflow.models.signature import infer_signature


mlflow.sklearn.autolog(log_input_examples=True,silent=True)



with mlflow.start_run() as run:
  
  skdtc_classifier = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5
  )

  model = Pipeline([
      ("classifier", skdtc_classifier),
  ])

 
  X_train = training_df.drop(['label'], axis=1)
  y_train = training_df['label'] = training_df['label'].astype(float)

  x_sample = X_train[0:2]
  y_sample = y_train[0:2]
  model.fit(X_train, y_train)
  input_example=X_train[0:2]

  fe.log_model(
    model=model,
    artifact_path="model",
    flavor=mlflow.sklearn,
    training_set=training_set,
    input_example=x_sample,
  )

  


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Use the model for inference

# COMMAND ----------

inference_df = spark.sql('select primary_account_number, merchant_id, date("2024-04-15") as observation_date, amount from fraud_eol where observation_date = "2024-04-15"')
display(inference_df)

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

logged_model = 'runs:/f6170ca6841842a7b86212715a9e0ba7/model'
#logged_model = f"runs:/{run.info.run_id}/model"

# This model was packaged by Feature Store.
# To retrieve features prior to scoring, call FeatureStoreClient.score_batch.
predictions = fs.score_batch(logged_model, inference_df)
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register With Unity Catalog

# COMMAND ----------


logged_model = 'runs:/f6170ca6841842a7b86212715a9e0ba7/model'
#logged_model = f"runs:/{run.info.run_id}/model"

mlflow.register_model(
  logged_model, f"{catalog}.{schema}.transaction_fraud"
)

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias(f"{catalog}.{schema}.transaction_fraud", "champion", 1)

# COMMAND ----------



model_version_uri = f"models:/{catalog}.{schema}.transaction_fraud@champion" 
predictions = fe.score_batch(
    model_uri=model_version_uri,
    df=inference_df
)
display(predictions)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Online Inference
# MAGIC
# MAGIC ### First the feature tables must be synced to an online table
# MAGIC
# MAGIC create online tables through UI or following api. see https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#create-an-online-table-using-apis
# MAGIC
# MAGIC
# MAGIC Use databricks online store or Cosmos DB: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/publish-features#cosmosdb-compatibility
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use api to create online tables

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as c


table_name = f"{catalog}.{schema}.transaction_aggregates"
w = WorkspaceClient()

def online_table_exists(table_name):
    w = WorkspaceClient()
    try:
        w.online_tables.get(name=table_name)
        return True
    except Exception as e:
        print(str(e))
        return 'already exists' in str(e)
    return False
def create_online_table(table_name, pks, timeseries_key=None):
    w = WorkspaceClient()
    online_table_name = table_name+"_online"
    if not online_table_exists(online_table_name):
        from databricks.sdk.service import catalog as c
        print(f"Creating online table for {online_table_name}...")
        spark.sql(f'ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')
        spec = c.OnlineTableSpec(source_table_full_name=table_name, primary_key_columns=pks, run_triggered={'triggered': 'true'}, timeseries_key=timeseries_key)
        w.online_tables.create(name=online_table_name, spec=spec)




create_online_table(f"{catalog}.{schema}.transaction_aggregates", ["customer_id"], "observation_date") 
create_online_table(f"{catalog}.{schema}.merchants", ["merchant_id"]) 
create_online_table(f"{catalog}.{schema}.customers", ["customer_id"], "observation_date") 



# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the model to a serving endpoint
# MAGIC
# MAGIC ####serve model using Unity Catalog UI
# MAGIC
# MAGIC or
# MAGIC
# MAGIC ####use workspace api

# COMMAND ----------

# MAGIC %md
# MAGIC ### Worskapce API for serving a model

# COMMAND ----------


from databricks.sdk.service.serving import ServedModelInput, ServedModelInputWorkloadSize, EndpointCoreConfigInput

endpoint_name = "bmac_cc_fraud_endpoint"
wc = WorkspaceClient()
served_models =[ServedModelInput(f"{catalog}.{schema}.transaction_fraud", model_version=9, workload_size=ServedModelInputWorkloadSize.SMALL, scale_to_zero_enabled=True)]
try:
    print(f'Creating endpoint {endpoint_name} with latest version...')
    wc.serving_endpoints.create_and_wait(endpoint_name, config=EndpointCoreConfigInput(served_models=served_models))
except Exception as e:
    if 'already exists' in str(e):
        print(f'Endpoint exists, updating with latest model version...')
        wc.serving_endpoints.update_config_and_wait(endpoint_name, served_models=served_models)
    else: 
        raise e

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Call model

# COMMAND ----------

data = [
    {
        "primary_account_number": 10048,
        "observation_date": "2024-04-15",
        "merchant_id": 20018,
        "amount": 465.25
    }
]
print('Data sent to the model:')
print(data)


inferences = wc.serving_endpoints.query(endpoint_name, inputs=data)
print(inferences.predictions)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import os

w = WorkspaceClient()

token = w.tokens.create(
    lifetime_seconds=3000
)
os.environ['DATABRICKS_TOKEN'] = token.token_value

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/bmac_cc_fraud_endpoint/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
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
# MAGIC SELECT ai_query('bmac_cc_fraud_endpoint',
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

fe.create_feature_spec(
  name="bmac.credit_card_fraud_demo.credit_fraud_features",
  features=feature_lookups,
)

# COMMAND ----------

from databricks.feature_engineering.entities.feature_serving_endpoint import (
  ServedEntity,
  EndpointCoreConfig,
)

fe.create_feature_serving_endpoint(
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
