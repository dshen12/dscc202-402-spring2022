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
from pyspark.sql import functions as F6

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
wallet_count_df.printSchema()

tokens_df.cache()
tokens_df.printSchema()


# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import dense_rank
wallet_count_df = wallet_count_df.withColumn("new_tokenId",dense_rank().over(Window.orderBy("token_address")))

wallet_count_df = wallet_count_df.withColumn("new_walletId",dense_rank().over(Window.orderBy("wallet_address")))

wallet_count_df = wallet_count_df.withColumnRenamed("token_address","tokenId")
wallet_count_df = wallet_count_df.withColumnRenamed("wallet_address","walletId")

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def prepareSubplot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0, subplots=(1, 1)):
    """Template for generating the plot layout."""
    plt.close()
    fig, axList = plt.subplots(subplots[0], subplots[1], figsize=figsize, facecolor='white',
                               edgecolor='white')
    if not isinstance(axList, np.ndarray):
        axList = np.array([axList])
        
    for ax in axList.flatten():
        ax.axes.tick_params(labelcolor='#999999', labelsize='10')
        for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
            axis.set_ticks_position('none')
            axis.set_ticks(ticks)
            axis.label.set_color('#999999')
            if hideLabels: axis.set_ticklabels([])
        ax.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
        map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
        
    if axList.size == 1:
        axList = axList[0]  # Just return a single axes object for a regular plot
    return fig, axList
    

from pyspark.sql import DataFrame
import inspect
def printDataFrames(verbose=False):
    frames = inspect.getouterframes(inspect.currentframe())
    notebookGlobals = frames[1][0].f_globals
    for k,v in notebookGlobals.items():
        if isinstance(v, DataFrame) and '_' not in k:
            print("{0}: {1}".format(k, v.columns)) if verbose else print("{0}".format(k))


def printLocalFunctions(verbose=False):
    frames = inspect.getouterframes(inspect.currentframe())
    notebookGlobals = frames[1][0].f_globals
    import types
    ourFunctions = [(k, v.__doc__) for k,v in notebookGlobals.items() if isinstance(v, types.FunctionType) and v.__module__ == '__main__']
    
    for k,v in ourFunctions:
        print("** {0} **".format(k))
        if verbose:
            print(v)

# COMMAND ----------

# count total entries
total_entries = wallet_count_df.count()

# find percentage listens by number of songs played
number_transactions = []
for i in range(10):
  number_transactions.append(float(wallet_count_df.filter(wallet_count_df.transaction == i+1).count())/total_entries*100)

# create bar plot
bar_width = 0.7
colorMap = 'Set1'
cmap = cm.get_cmap(colorMap)

fig, ax = prepareSubplot(np.arange(0, 10, 1), np.arange(0, 80, 5))
plt.bar(np.linspace(1,10,10), number_transactions, width=bar_width, color=cmap(0))
plt.xticks(np.linspace(1,10,10) + bar_width/2.0, np.linspace(1,10,10))
plt.xlabel('Number of Plays'); plt.ylabel('%')
plt.title('Percentage Number of Plays of Songs')
display(fig)

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
df=sqlContext.sql("select * from G05_db.SilverTable_Wallets")
result_df = df.select("*").toPandas()
token_ids_with_total_transactions = wallet_count_df.groupBy('tokenId') \
                                                       .agg(F.count(wallet_count_df.transaction).alias('Transaction_Count'),
                                                            F.sum(wallet_count_df.transaction).alias('Total_transaction')) \
                                                       .orderBy('Total_transaction', ascending = False)

print('token_ids_with_total_transactions:',
token_ids_with_total_transactions.show(3, truncate=False))

# Join with metadata to get artist and song title
token_names_with_transaction_df = token_ids_with_total_transactions.join(tokens_df,  token_ids_with_total_transactions.tokenId == tokens_df.address) \
                                                      .filter('Transaction_Count >= 2') \
                                                      .select('name', 'symbol', 'address', 'Transaction_Count','Total_transaction') \
                                                      .orderBy('Total_transaction', ascending = False)

#print('token_names_with_transaction_df:',
#token_names_with_transaction_df.show(20, truncate = False))

# COMMAND ----------

seed = 42
(split_60_df, split_a_20_df, split_b_20_df) = wallet_count_df.randomSplit([0.6, 0.2, 0.2], seed = seed)

# Let's cache these datasets for performance
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()

# COMMAND ----------

validation_df = validation_df.withColumn("transaction", validation_df["transaction"].cast(DoubleType()))

# COMMAND ----------

modelName = "G06_Model"
def mlflow_als(rank,maxIter,regParam):
  with mlflow.start_run(run_name = modelName+"-run") as run:
    seed = 42
    (split_60_df, split_a_20_df, split_b_20_df) = wallet_count_df.randomSplit([0.6, 0.2, 0.2], seed = seed)
    training_df = split_60_df.cache()
    validation_df = split_a_20_df.cache()
    test_df = split_b_20_df.cache()
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
    
    # Log model
    mlflow.spark.log_model(spark_model=alsModel, signature = signature,artifact_path='als-model',registered_model_name=modelName)
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
    print("ALS model with run_id", {runID},"and experiment_id",{experimentID})

# COMMAND ----------

mlflow_als(5,5,0.6)
#mlflow_als(5,10,0.2)
#mlflow_als(5,5,0.3)
#mlflow_als(5,10,0.3)

# COMMAND ----------

client = MlflowClient()
model_versions = []
    
for mv in client.search_model_versions(f"name='{modelName}'"):
    model_versions.append(dict(mv)['version'])
    if dict(mv)['current_stage'] == 'Staging':
        print("Archiving: {}".format(dict(mv)))
        # Archive the currently staged model
        client.transition_model_version_stage(
            name= modelName,
            version=dict(mv)['version'],
            stage="Archived"
        )

# COMMAND ----------

client.transition_model_version_stage(name=modelName,version=model_versions[0],stage="Staging")

# COMMAND ----------

runs = client.search_runs('3805545008156543', max_results=1)
runs[0].data.metrics

# COMMAND ----------



# COMMAND ----------

import uuid

model_name = "ALS_Recommendation"
model_name

# COMMAND ----------

runID = '6ad23faf96e54547890957f2a41555ac'
model_uri = "runs:/{run_id}/als-model".format(run_id=runID)

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Production')

# COMMAND ----------

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

# COMMAND ----------

mlflow.spark.load_model('models:/'+modelName+'/Staging')

# COMMAND ----------

mlflow.spark.load_model('models:/'+'als_model_4912847e91'+'/Production')

# COMMAND ----------

model = mlflow.spark.load_model('models:/'+modelName+'/Staging')

# COMMAND ----------

test_predictions = model.transform(test_df)

# COMMAND ----------

reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="buy_count", metricName="rmse")
RMSE = reg_eval.evaluate(test_predictions)

# COMMAND ----------

test_predictions.withColumn("prediction", test_predictions["prediction"].cast(IntegerType()))

# COMMAND ----------

test_predictions

# COMMAND ----------

def rd(df):
    df

# COMMAND ----------

model

# COMMAND ----------

RMSE

# COMMAND ----------

test_predictions.show()

# COMMAND ----------

model = mlflow.spark.load_model('models:/'+'als_model_4912847e91'+'/Production')
test_predictions_prod = model.transform(test_df)
RMSE_prod = reg_eval.evaluate(test_predictions_prod)

# COMMAND ----------

RMSE_prod

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
