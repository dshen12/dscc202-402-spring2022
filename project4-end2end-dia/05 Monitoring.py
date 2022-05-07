# Databricks notebook source
# MAGIC %md
# MAGIC ## Rubric for this module
# MAGIC - Implement a routine to "promote" your model at **Staging** in the registry to **Production** based on a boolean flag that you set in the code.
# MAGIC - Using wallet addresses from your **Staging** and **Production** model test data, compare the recommendations of the two models.

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
# MAGIC ## Your Code Starts Here...

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql import functions as F6
from pyspark.sql.functions import col
# Load the data from the lake
#wallet_balance_df = spark.read.format('delta').load("/mnt/dscc202-datasets/misc/G06/tokenrec/newtables/")
wallet_count_df = spark.read.format('delta').load("/mnt/dscc202-datasets/misc/G06/tokenrec/wctables/")
tokens_df = spark.read.format('delta').load("/mnt/dscc202-datasets/misc/G06/tokenrec/tokentables/")
#unique_wallets = wallet_count_df.select('wallet_address').distinct().count()
#unique_tokens = wallet_count_df.select('token_address').distinct().count()
#print('Number of unique wallets: {0}'.format(unique_wallets))
#print('Number of unique tokens: {0}'.format(unique_tokens))
wallet_count_df.cache()
tokens_df.cache()

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import dense_rank
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

wallet_count_df = wallet_count_df.withColumn("new_tokenId",dense_rank().over(Window.orderBy("token_address")))
wallet_count_df = wallet_count_df.withColumn("new_walletId",dense_rank().over(Window.orderBy("wallet_address")))
wallet_count_df = wallet_count_df.withColumnRenamed("token_address","tokenId")
wallet_count_df = wallet_count_df.withColumnRenamed("wallet_address","walletId")

seed = 42
(split_60_df, split_a_20_df, split_b_20_df) = wallet_count_df.randomSplit([0.6, 0.2, 0.2], seed = seed)
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()
validation_df = validation_df.withColumn("buy_count", validation_df["buy_count"].cast(DoubleType()))
test_df = test_df.withColumn("buy_count", test_df["buy_count"].cast(DoubleType()))

# COMMAND ----------

from pyspark.ml.recommendation import ALS
import uuid
modelName = "G06_Model_Monitoring"
client = MlflowClient()
def mlflow_als(rank,maxIter,regParam):
    with mlflow.start_run(run_name = modelName) as run:
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
        
        mlflow.spark.log_model(spark_model=alsModel, signature = signature,
                                 artifact_path='als-model',registered_model_name=modelName)
        print("ALS model with run_id", {runID},"and experiment_id",{experimentID})
        
        return runID,experimentID

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boolean Flag: Stop retrain model when rmse diminishes to 680

# COMMAND ----------

rmse=814 #Baseline RMSE
rank = 25
maxIter=25
regParam=0.6

# get above matrics from notebook 4, in order to reduce the run time, we directly input values

runs=[]
experiments=[]
version = 0
while rmse >= 680: ##boolean flag: stop retrain model when rmse reach 600
    version+=1
    run_id,experiment_id=mlflow_als(rank,maxIter,regParam)
    runs.append(run_id)
    experiments.append(experiment_id)
    client.transition_model_version_stage(name= modelName, version=version, stage="Staging")
    runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
    rmse = runs[0].data.metrics['valid_rmse']
    regParam +=0.1
client.transition_model_version_stage(name= modelName, version=version, stage="Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use test dataset for Staging Model

# COMMAND ----------

staging_model = mlflow.spark.load_model('models:/'+modelName+'/Staging')

# COMMAND ----------

test_predictions_staging = staging_model.transform(test_df)
test_predictions_staging = test_predictions_staging.withColumn("prediction", F.round(col("prediction"),0))

# COMMAND ----------

reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="buy_count", metricName="rmse")
print("Test RMSE from Staging Model is:", reg_eval.evaluate(test_predictions_staging))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use test dataset for Production Model

# COMMAND ----------

production_model = mlflow.spark.load_model('models:/'+modelName+'/Production')
test_predictions_production = staging_model.transform(test_df)
test_predictions_production = test_predictions_production.withColumn("prediction", F.round(col("prediction"),0))

# COMMAND ----------

reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="buy_count", metricName="rmse")
print("Test RMSE from Production Model is:", reg_eval.evaluate(test_predictions_production))

# COMMAND ----------

# After we round the results to the nearest integer, they have same RMSE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top 5 Recommendation using Staging Model

# COMMAND ----------

from pyspark.sql import functions as F
wallet_address = Utils.create_widgets()[0]
dff = wallet_count_df.where(col("walletId") == wallet_address).select('new_walletId')
walletId = dff.collect()[0]['new_walletId']

# COMMAND ----------

def recommend(walletId,model):
    assert str(type(model))=="<class 'pyspark.ml.pipeline.PipelineModel'>"
    bought_token = wallet_count_df.filter(wallet_count_df.new_walletId == walletId).join(tokens_df, wallet_count_df.tokenId == tokens_df.address).select('new_tokenId', 'symbol', 'name','buy_count','tokenId')
    unbought_token = wallet_count_df.filter(~ wallet_count_df['new_tokenId'].isin([token['new_tokenId'] for token in bought_token.collect()])).select('new_tokenId').withColumn('new_walletId', F.lit(walletId)).distinct()
 
    
    predicted_buy_counts = model.transform(unbought_token)
 
    return (bought_token.select('symbol','name','buy_count','tokenId').orderBy('buy_count', ascending = False), predicted_buy_counts.join(wallet_count_df, 'new_tokenId') \
                .join(tokens_df, wallet_count_df.tokenId == tokens_df.address) \
                .select('symbol', 'name', 'prediction','tokenId') \
                .distinct() \
                .orderBy('prediction', ascending = False)
                .limit(5))

# COMMAND ----------

stage_model = mlflow.spark.load_model('models:/'+modelName+'/Staging')
stage_1,stage_5= recommend(walletId,stage_model)
stage_5 = stage_5.withColumn("prediction", F.round(col("prediction"),0))
stage_5.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top 5 Recommendation using Production Model

# COMMAND ----------

product_model = mlflow.spark.load_model('models:/'+modelName+'/Production')
product_1,product_5= recommend(walletId,product_model)
product_5 = product_5.withColumn("prediction",F.round(col("prediction"),0))
product_5.show()

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
