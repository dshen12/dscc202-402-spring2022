# Databricks notebook source
# MAGIC %md
# MAGIC ## Token Recommendation
# MAGIC <table border=0>
# MAGIC   <tr><td><img src='https://data-science-at-scale.s3.amazonaws.com/images/rec-application.png'></td>
# MAGIC     <td>Your application should allow a specific wallet address to be entered via a widget in your application notebook.  Each time a new wallet address is entered, a new recommendation of the top tokens for consideration should be made. <br> **Bonus** (3 points): include links to the Token contract on the blockchain or etherscan.io for further investigation.</td></tr>
# MAGIC   </table>

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
# MAGIC ## Your code starts here...

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

from pyspark.sql import functions as F
wallet_address = Utils.create_widgets()[0]
dff = wallet_count_df.where(col("walletId") == wallet_address).select('new_walletId')
assert dff.count()>0, "No such user"
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

client = MlflowClient()

for mdl in client.search_registered_models():
    model_name=dict(mdl)['name']
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if dict(mv)['current_stage']=='Production':
            modelName=dict(mv)['name']
            
product_model = mlflow.spark.load_model('models:/'+modelName+'/Production')

# COMMAND ----------

product_1,product_5= recommend(walletId,product_model)

# COMMAND ----------

product_5 = product_5.withColumn("prediction", F.round(col("prediction"),0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Recommendation with LINK to etherscan.io

# COMMAND ----------

result=product_5.withColumn('website', F.lit('https://etherscan.io/address/')).select("symbol","name","prediction",concat(col("website"), col("tokenId")).alias('link'))
result.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show Recommendation UI

# COMMAND ----------

head=f"""
<html>
<head>
<meta charset="utf-8">
<title>Token Recommendation(runoob.com)</title>
</head>
<body>
    <h1>Recommend tokens for user address:</h1>
    <p>{wallet_address}</p>
    <table boder="0"cellspacing="5" cellpadding="10" bgcolor="gold" class="tabtop13" style="width:50%" >
  <tr>
    <tr>
        <th style="width:30%">Name</th>
        <th style="width:30%">Symbol</th>
        <th style="width:30%">Link</th>
    </tr>
    """
end=f"""
</table>
</body>
</html>
"""
htmltext=head
for c in result.collect():
    tr=f"""
    <tr>
        <td style="text-align:center">{c['name']}</td>
        <td style="text-align:center">{c['symbol']}</td>
        <td style="text-align:center"><a href="{c['link']}">link</a> </td>
    </tr>
    """
    htmltext+=tr

# COMMAND ----------

displayHTML(htmltext)

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
