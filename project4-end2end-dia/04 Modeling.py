# Databricks notebook source
# MAGIC %md
# MAGIC ## Rubric for this module
# MAGIC - Using the silver delta table(s) that were setup by your ETL module train and validate your token recommendation engine. Split, Fit, Score, Save
# MAGIC - Log all experiments using mlflow
# MAGIC - capture model parameters, signature, training/test metrics and artifacts
# MAGIC - Tune hyperparameters using an appropriate scaling mechanism for spark.  [Hyperopt/Spark Trials ](https://docs.databricks.com/_static/notebooks/hyperopt-spark-ml.html)
# MAGIC - Register your best model from the training run at **Staging**.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Your Code starts here...

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql import functions as F

# Load the data from the lake
wallet_balance_df = spark.read.format('delta').load("/mnt/dscc202-datasets/misc/G06/tokenrec/newtables")
wallet_count_df = spark.read.format('delta').load("/mnt/dscc202-datasets/misc/G06/tokenrec/wctables")
tokens_df = spark.read.format('delta').load("/mnt/dscc202-datasets/misc/G06/tokenrec/tokentables")


# COMMAND ----------

from pyspark.sql.functions import col

wallet_count_df = wallet_count_df.withColumn("transaction", col("buy_count") + col("sell_count"))

# COMMAND ----------

unique_wallets = wallet_count_df.select('wallet_address').distinct().count()
unique_tokens = wallet_count_df.select('token_address').distinct().count()
print('Number of unique wallets: {0}'.format(unique_wallets))
print('Number of unique tokens: {0}'.format(unique_tokens))

# COMMAND ----------

# cache
wallet_count_df.cache()
tokens_df.cache()

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import dense_rank
wallet_count_df = wallet_count_df.withColumn("new_tokenId",dense_rank().over(Window.orderBy("token_address")))
wallet_count_df = wallet_count_df.withColumn("new_walletId",dense_rank().over(Window.orderBy("wallet_address")))

wallet_count_df = wallet_count_df.withColumnRenamed("token_address","tokenId")
wallet_count_df = wallet_count_df.withColumnRenamed("wallet_address","walletId")

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F
from delta.tables import *
import random

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

seed = 42
(split_60_df, split_a_20_df, split_b_20_df) = wallet_count_df.randomSplit([0.6, 0.2, 0.2], seed = seed)

# Let's cache these datasets for performance
training_df = split_60_df
validation_df = split_a_20_df
test_df = split_b_20_df

# COMMAND ----------

validation_df = validation_df.withColumn("buy_count", validation_df["buy_count"].cast(DoubleType()))
test_df = test_df.withColumn("buy_count", test_df["buy_count"].cast(DoubleType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline

# COMMAND ----------

modelName = "G06_Model"
def mlflow_als(rank,maxIter,regParam):

    with mlflow.start_run(run_name = modelName+"-run", nested=True) as run:
        seed = 42
        (split_60_df, split_a_20_df, split_b_20_df) = wallet_count_df.randomSplit([0.6, 0.2, 0.2], seed = seed)
        training_df = split_60_df.cache()
        validation_df = split_a_20_df.cache()
        test_df = split_b_20_df.cache()
        validation_df = validation_df.withColumn("buy_count", validation_df["buy_count"].cast(DoubleType()))
        test_df = test_df.withColumn("buy_count", test_df["buy_count"].cast(DoubleType()))
        input_schema = Schema([ColSpec("integer", "new_tokenId"),ColSpec("integer", "new_walletId")])
        output_schema = Schema([ColSpec("double")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
        # Create model
        # Initialize our ALS learner
        als = ALS(rank=rank, maxIter=maxIter, regParam=regParam,seed=42)
        als.setItemCol("new_tokenId")\
           .setRatingCol("buy_count")\
           .setUserCol("new_walletId")\
           .setColdStartStrategy("drop")
        reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="buy_count", metricName="rmse")

        alsModel = als.fit(training_df)
        validation_metric = reg_eval.evaluate(alsModel.transform(validation_df))
    
        mlflow.log_metric('valid_' + reg_eval.getMetricName(), validation_metric) 
        mlflow.log_param("rank", rank)
        mlflow.log_param("maxIter", maxIter)
        mlflow.log_param("regParam", regParam)
        
        runID = run.info.run_uuid
        experimentID = run.info.experiment_id
    
        # Log model
        mlflow.spark.log_model(spark_model=alsModel, signature = signature,artifact_path='als-model')
    return alsModel, validation_metric

# COMMAND ----------

initial_model, val_metric = mlflow_als(rank = 20,maxIter = 20, regParam = 0.5)
print(f"The trained ALS achieved an RMSE {val_metric} on the validation data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyper Tuning Using Hyperopt fmin()

# COMMAND ----------

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
 
def train_with_hyperopt(params):
    rank = int(params['rank'])
    maxIter = int(params['maxIter'])
    regParam= float(params['regParam'])
    model, rmse = mlflow_als(rank, maxIter, regParam)[0], mlflow_als(rank, maxIter, regParam)[1]
    loss = rmse
    return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

import numpy as np
space = {
  'rank': hp.choice('rank', [20, 25]),
  'maxIter': hp.choice('maxIter', [20, 25]),
  'regParam': hp.choice('regParam', [0.5, 0.6]),
}

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

algo=tpe.suggest
 
with mlflow.start_run() as run:
    best_params = fmin(fn=train_with_hyperopt,space=space,algo=algo,max_evals=3)

# COMMAND ----------

from hyperopt import space_eval
best = space_eval(space,best_params)

# COMMAND ----------

best_rank = best['rank']
best_iter = best['maxIter']
best_reg = best['regParam']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain the full train set with best param and register to staging

# COMMAND ----------

final_model, validation_rmse = mlflow_als(best_rank, best_iter, best_reg)

# COMMAND ----------

client = MlflowClient()
model_versions = []


for mv in client.search_model_versions(f"name='{modelName}'"):
    model_versions.append(dict(mv)['version'])
    if dict(mv)['current_stage'] == 'Staging':
        print("Archiving: {}".format(dict(mv)))
        client.transition_model_version_stage(
            name=modelName,
            version=dict(mv)['version'],
            stage="Archived"
        )
# best model
client.transition_model_version_stage(name=modelName,version=model_versions[0],stage="Staging")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test dataset

# COMMAND ----------

test_predictions = final_model.transform(test_df)
test_predictions = test_predictions.withColumn("prediction", F.round(col("prediction"),0))

# COMMAND ----------

test_predictions.show(5)

# COMMAND ----------

reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="buy_count", metricName="rmse")
reg_eval.evaluate(test_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recommendation

# COMMAND ----------

from pyspark.sql import functions as F
def recommend(walletId: int)->(DataFrame,DataFrame):
    bought_token = wallet_count_df.filter(wallet_count_df.new_walletId == walletId).join(tokens_df, wallet_count_df.tokenId == tokens_df.address).select('new_tokenId', 'symbol', 'name','buy_count')
    unbought_token = wallet_count_df.filter(~ wallet_count_df['new_tokenId'].isin([token['new_tokenId'] for token in bought_token.collect()])).select('new_tokenId').withColumn('new_walletId', F.lit(walletId)).distinct()
 
    model = mlflow.spark.load_model('models:/'+modelName+'/Staging')
    predicted_buy_counts = model.transform(unbought_token)
 
    return (bought_token.select('symbol','name','buy_count').orderBy('buy_count', ascending = False), predicted_buy_counts.join(wallet_count_df, 'new_tokenId') \
                .join(tokens_df, wallet_count_df.tokenId == tokens_df.address) \
                .select('symbol', 'name', 'prediction') \
                .distinct() \
                .orderBy('prediction', ascending = False)
                .limit(5))

# COMMAND ----------

modelName = "G06_Model"
wallet_address = Utils.create_widgets()[0]
dff = wallet_count_df.where(col("walletId") == wallet_address).select('new_walletId')
walletId = dff.head()[0]
a = recommend(walletId)
top_5_recommend_token = a[1]
top_5_recommend_token = top_5_recommend_token.select('symbol', 'name')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top 5 token for wallet address: 0x000000cce580fb2b76b2d1196c237db199d82505

# COMMAND ----------

top_5_recommend_token.show()

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
