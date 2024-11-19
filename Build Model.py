# Databricks notebook source
catalog = 'bmac'
schema = 'credit_card_fraud_demo'

# COMMAND ----------

spark.sql(f"use {catalog}.{schema}")

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
# MAGIC CREATE or replace FUNCTION bmac.credit_card_fraud_demo.distance(from_zip string, to_zip string)
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
fe = FeatureEngineeringClient()

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

eol_df = spark.sql('select * from fraud_eol')

training_set = fe.create_training_set(
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
from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")
mlflow.sklearn.autolog(log_input_examples=True,silent=True)


skdtc_classifier = DecisionTreeClassifier(
  criterion="gini",
  max_depth=10
)

model = Pipeline([
    ("classifier", skdtc_classifier),
])

training_df = training_df.toPandas()
X_train = training_df.drop(['label'], axis=1)
y_train = training_df['label'] = training_df['label'].astype(float)

x_sample = X_train[0:2]
y_sample = y_train[0:2]

with mlflow.start_run():
  model.fit(X_train, y_train)
  training_set=training_set, 
  input_example=X_train[0:2]
  signature=infer_signature(x_sample, y_sample), # schema of the dataset, not necessary with FS, 






# COMMAND ----------

# MAGIC %md
# MAGIC ### Let use the model for Inference
# MAGIC
# MAGIC Note that I need to supply all of the data.  The model stored in mflow is just a regular sklearn model.  There is nothing to connect it to the feature store.

# COMMAND ----------

import pandas as pd

logged_model = 'runs:/5e024a5f097a4bfdb016b15e434d2fe6/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

data = {
    "columns": [
        "primary_account_number",
        "amount",
        "14_day_total",
        "7_day_total",
        "7_day_transaction_count",
        "merchant_distance_from_home"
    ],
    "data": [
        [10000, 132.12, 1564.62, 386.88, 1, 240],
        [10001, 78.11, 2752.53, 1574.79, 2, 97],
        [10002, 475.33, 2142.27, 1755.39, 2, 230],
        [10003, 419.24, 3805.8, 2231.01, 3, 47],
        [10004, 286.99, 3135.03, 1711.86, 3, 135]
    ]
}

# Create the DataFrame
df = pd.DataFrame(data['data'], columns=data['columns']).astype({
    'primary_account_number': 'int64',
    'amount': 'float32',
    '14_day_total': 'float64',
    '7_day_total': 'float64',
    '7_day_transaction_count': 'int64',
    'merchant_distance_from_home': 'float64'
})


loaded_model.predict(pd.DataFrame(df))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's make the connection between the model and the feature store by using the fe.log_model api call
# MAGIC
# MAGIC note that this will create a new run in mlflow

# COMMAND ----------

run_id = '5e024a5f097a4bfdb016b15e434d2fe6'
logged_modeld_model = f"runs:/{run_id}/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)

x_sample = eol_df.toPandas()[0:3]

fe.log_model(
              model=loaded_model, # object of your model
              artifact_path="model",
              flavor=mlflow.sklearn, 
              training_set=training_set, 
              input_example=x_sample, # example of the dataset, should be Pandas 
              registered_model_name=f"{catalog}.{schema}.transaction_fraud"
          )

metrics = mlflow.get_run(run_id).data.metrics
params = mlflow.get_run(run_id).data.params
mlflow.log_metrics(metrics)
mlflow.log_params(params)
mlflow.set_tag(key='feature_store', value='churn_model_demo')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Let's use the model for inference

# COMMAND ----------

inference_df = spark.sql('select primary_account_number, merchant_id, date("2024-04-15") as observation_date, amount from fraud_eol where observation_date = "2024-04-15"')
display(inference_df)

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias(f"{catalog}.{schema}.transaction_fraud", "champion", 7)

# COMMAND ----------

#note that the URI is made up of current catalog and schema and the name of the model from fs.log_model above. 

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

def create_online_table(table_name, primary_key_columns, timeseries_key=None):
    if not online_table_exists(table_name):
        spark.sql(f'ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')
        online_table_name = table_name+"_online"
        spec = c.OnlineTableSpec(source_table_full_name=table_name, primary_key_columns=primary_key_columns, run_triggered={'triggered': 'true'}, timeseries_key=timeseries_key)
        w.online_tables.create(name=online_table_name, spec=spec)


create_online_table(f"{catalog}.{schema}.transaction_aggregates", ["customer_id"], "observation_date") 
create_online_table(f"{catalog}.{schema}.merchants", ["merchant_id"]) 
create_online_table(f"{catalog}.{schema}.customers", ["customer_id"], "observation_date") 



# COMMAND ----------

# MAGIC %md
# MAGIC ### Then deploy the model to a serving endpoint
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


endpoint_name = "bmac_cc_fraud_endpoint"
wc = WorkspaceClient()
served_models =[ServedModelInput(model_full_name, model_version=latest_model.version, workload_size=ServedModelInputWorkloadSize.SMALL, scale_to_zero_enabled=True)]
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

# MAGIC %md
# MAGIC Deploy model to serving endpoint through UI or code (reference?)

# COMMAND ----------



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
    url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/credit_card_fraud/invocations'
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
