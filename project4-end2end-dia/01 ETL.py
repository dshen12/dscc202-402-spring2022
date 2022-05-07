# Databricks notebook source
# MAGIC %md
# MAGIC ## Ethereum Blockchain Data Analysis - <a href=https://github.com/blockchain-etl/ethereum-etl-airflow/tree/master/dags/resources/stages/raw/schemas>Table Schemas</a>
# MAGIC - **Transactions** - Each block in the blockchain is composed of zero or more transactions. Each transaction has a source address, a target address, an amount of Ether transferred, and an array of input bytes. This table contains a set of all transactions from all blocks, and contains a block identifier to get associated block-specific information associated with each transaction.
# MAGIC - **Blocks** - The Ethereum blockchain is composed of a series of blocks. This table contains a set of all blocks in the blockchain and their attributes.
# MAGIC - **Receipts** - the cost of gas for specific transactions.
# MAGIC - **Traces** - The trace module is for getting a deeper insight into transaction processing. Traces exported using <a href=https://openethereum.github.io/JSONRPC-trace-module.html>Parity trace module</a>
# MAGIC - **Tokens** - Token data including contract address and symbol information.
# MAGIC - **Token Transfers** - The most popular type of transaction on the Ethereum blockchain invokes a contract of type ERC20 to perform a transfer operation, moving some number of tokens from one 20-byte address to another 20-byte address. This table contains the subset of those transactions and has further processed and denormalized the data to make it easier to consume for analysis of token transfer events.
# MAGIC - **Contracts** - Some transactions create smart contracts from their input bytes, and this smart contract is stored at a particular 20-byte address. This table contains a subset of Ethereum addresses that contain contract byte-code, as well as some basic analysis of that byte-code. 
# MAGIC - **Logs** - Similar to the token_transfers table, the logs table contains data for smart contract events. However, it contains all log data, not only ERC20 token transfers. This table is generally useful for reporting on any logged event type on the Ethereum blockchain.
# MAGIC 
# MAGIC In Addition, there is a price feed that changes daily (noon) that is in the **token_prices_usd** table
# MAGIC 
# MAGIC ### Rubric for this module
# MAGIC - Transform the needed information in ethereumetl database into the silver delta table needed by your modeling module
# MAGIC - Clearly document using the notation from [lecture](https://learn-us-east-1-prod-fleet02-xythos.content.blackboardcdn.com/5fdd9eaf5f408/8720758?X-Blackboard-Expiration=1650142800000&X-Blackboard-Signature=h%2FZwerNOQMWwPxvtdvr%2FmnTtTlgRvYSRhrDqlEhPS1w%3D&X-Blackboard-Client-Id=152571&response-cache-control=private%2C%20max-age%3D21600&response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%27Delta%2520Lake%2520Hands%2520On%2520-%2520Introduction%2520Lecture%25204.pdf&response-content-type=application%2Fpdf&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAAaCXVzLWVhc3QtMSJHMEUCIQDEC48E90xPbpKjvru3nmnTlrRjfSYLpm0weWYSe6yIwwIgJb5RG3yM29XgiM%2BP1fKh%2Bi88nvYD9kJNoBNtbPHvNfAqgwQIqP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgw2MzU1Njc5MjQxODMiDM%2BMXZJ%2BnzG25TzIYCrXAznC%2BAwJP2ee6jaZYITTq07VKW61Y%2Fn10a6V%2FntRiWEXW7LLNftH37h8L5XBsIueV4F4AhT%2Fv6FVmxJwf0KUAJ6Z1LTVpQ0HSIbvwsLHm6Ld8kB6nCm4Ea9hveD9FuaZrgqWEyJgSX7O0bKIof%2FPihEy5xp3D329FR3tpue8vfPfHId2WFTiESp0z0XCu6alefOcw5rxYtI%2Bz9s%2FaJ9OI%2BCVVQ6oBS8tqW7bA7hVbe2ILu0HLJXA2rUSJ0lCF%2B052ScNT7zSV%2FB3s%2FViRS2l1OuThecnoaAJzATzBsm7SpEKmbuJkTLNRN0JD4Y8YrzrB8Ezn%2F5elllu5OsAlg4JasuAh5pPq42BY7VSKL9VK6PxHZ5%2BPQjcoW0ccBdR%2Bvhva13cdFDzW193jAaE1fzn61KW7dKdjva%2BtFYUy6vGlvY4XwTlrUbtSoGE3Gr9cdyCnWM5RMoU0NSqwkucNS%2F6RHZzGlItKf0iPIZXT3zWdUZxhcGuX%2FIIA3DR72srAJznDKj%2FINdUZ2s8p2N2u8UMGW7PiwamRKHtE1q7KDKj0RZfHsIwRCr4ZCIGASw3iQ%2FDuGrHapdJizHvrFMvjbT4ilCquhz4FnS5oSVqpr0TZvDvlGgUGdUI4DCdvOuSBjqlAVCEvFuQakCILbJ6w8WStnBx1BDSsbowIYaGgH0RGc%2B1ukFS4op7aqVyLdK5m6ywLfoFGwtYa5G1P6f3wvVEJO3vyUV16m0QjrFSdaD3Pd49H2yB4SFVu9fgHpdarvXm06kgvX10IfwxTfmYn%2FhTMus0bpXRAswklk2fxJeWNlQF%2FqxEmgQ6j4X6Q8blSAnUD1E8h%2FBMeSz%2F5ycm7aZnkN6h0xkkqQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220416T150000Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21600&X-Amz-Credential=ASIAZH6WM4PLXLBTPKO4%2F20220416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=321103582bd509ccadb1ed33d679da5ca312f19bcf887b7d63fbbb03babae64c) how your pipeline is structured.
# MAGIC - Your pipeline should be immutable
# MAGIC - Use the starting date widget to limit how much of the historic data in ethereumetl database that your pipeline processes.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)
spark.conf.set('wallet.address',wallet_address)
spark.conf.set('start.date',start_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## YOUR SOLUTION STARTS HERE...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Group 6 Pipeline
# MAGIC ![Image](https://github.com/carolxueyq/dscc202-402-spring2022/blob/final_project/project4-end2end-dia/pipeline.png?raw=true)

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables in ethereumetl

# COMMAND ----------

blockDF = spark.sql("select * from ethereumetl.blocks")
transactionDF = spark.sql("select * from ethereumetl.transactions")
receiptDF = spark.sql("select * from ethereumetl.receipts")

# COMMAND ----------

contractDF = spark.sql("select * from ethereumetl.contracts")
silvercontractDF = spark.sql("select * from ethereumetl.silver_contracts")
tokenDF = spark.sql("select * from ethereumetl.tokens")
tokentransferDF = spark.sql("select * from ethereumetl.token_transfers")
tokenpriceDF = spark.sql("select * from ethereumetl.token_prices_usd")

# COMMAND ----------

from pyspark.sql.functions import col,count

# COMMAND ----------

erc20_contract = silvercontractDF.filter(silvercontractDF["is_erc20"]==True)

# COMMAND ----------

from pyspark.sql.types import _parse_datatype_string
from pyspark.sql.types import *

ddl_schema = StructType([
  StructField("address", StringType(), True),
  StructField("bytecode", StringType(), True),
  StructField("is_erc20", StringType(), True),
  StructField("is_erc721", StringType(), True),   
])

assert ddl_schema == _parse_datatype_string("address string, bytecode string, is_erc20 string, is_erc721 string"), "File not present in Silver Path"
print("Assertion passed.")

# COMMAND ----------

dbutils.fs.rm(f"/mnt/dscc202-datasets/misc/G06/tokenrec/ectables/", recurse=True)
erc20_contract_delta_dir = f"/mnt/dscc202-datasets/misc/G06/tokenrec/ectables/"
dbutils.fs.mkdirs(erc20_contract_delta_dir)

# COMMAND ----------

erc20_contract.write.format("delta").mode("overwrite").save(erc20_contract_delta_dir)

# COMMAND ----------

spark.sql(
    """
DROP TABLE IF EXISTS g06_db.erc20_contract
"""
)
spark.sql(
    f"""
CREATE TABLE g06_db.erc20_contract
USING DELTA
LOCATION "{erc20_contract_delta_dir}"
"""
)

# COMMAND ----------

erc20_token_transfer = tokentransferDF.join(silvercontractDF, tokentransferDF.token_address==silvercontractDF.address,how="left").filter(silvercontractDF["is_erc20"]==True).drop("address","bytecode","is_erc721")

# COMMAND ----------

ddl_schema = StructType([
  StructField("token_address", StringType(), True),
  StructField("from_address", StringType(), True),
  StructField("to_address", StringType(), True),
  StructField("value", DecimalType(38,0), True),   
  StructField("transaction_hash", StringType(), True),
  StructField("log_index", LongType(), True),
  StructField("block_number", LongType(), True),
  StructField("start_block", LongType(), True),
  StructField("end_block", LongType(), True), 
  StructField("is_erc20", StringType(), True),
])

assert ddl_schema == _parse_datatype_string("token_address string ,from_address string ,to_address string ,value decimal(38,0) ,transaction_hash string ,log_index long ,block_number long ,start_block long ,end_block long ,is_erc20 string"), "File not present in Silver Path"

print("Assertion passed.")

# COMMAND ----------

dbutils.fs.rm(f"/mnt/dscc202-datasets/misc/G06/tokenrec/ettables/", recurse=True)
token_transfer_delta_dir = f"/mnt/dscc202-datasets/misc/G06/tokenrec/ettables/"
dbutils.fs.mkdirs(token_transfer_delta_dir)

# COMMAND ----------

erc20_token_transfer.write.format("delta").mode("overwrite").save(token_transfer_delta_dir)

# COMMAND ----------

spark.sql(
    """
DROP TABLE IF EXISTS g06_db.erc20_token_transfer
"""
)
spark.sql(
    f"""
CREATE TABLE g06_db.erc20_token_transfer
USING DELTA
LOCATION "{token_transfer_delta_dir}"
"""
)

# COMMAND ----------

erc20_tokentransferDF = spark.sql("select * from G06_db.erc20_token_transfer")

# COMMAND ----------

dbutils.fs.rm(f"/mnt/dscc202-datasets/misc/G06/tokenrec/tokentables/", recurse=True)
token_delta_dir = f"/mnt/dscc202-datasets/misc/G06/tokenrec/tokentables/"
dbutils.fs.mkdirs(token_delta_dir)

# COMMAND ----------

ddl_schema = StructType([
  StructField("address", StringType(), True),
  StructField("symbol", StringType(), True),
  StructField("name", StringType(), True),
  StructField("decimals", LongType(), True),
  StructField("total_supply", DecimalType(38,0), True),   
  StructField("start_block", LongType(), True),
  StructField("end_block", LongType(), True), 
])

assert ddl_schema == _parse_datatype_string("address string, symbol string, name string, decimals long, total_supply decimal(38,0), start_block long, end_block long"), "File not present in Silver Path"

print("Assertion passed.")

# COMMAND ----------

tokenDF.write.format("delta").mode("overwrite").save(token_delta_dir)

# COMMAND ----------

spark.sql(
    """
DROP TABLE IF EXISTS g06_db.token
"""
)
spark.sql(
    f"""
CREATE TABLE g06_db.token
LOCATION "{token_delta_dir}"
"""
)

# COMMAND ----------

dbutils.fs.rm(f"/mnt/dscc202-datasets/misc/G06/tokenrec/tokenpricetables/", recurse=True)
token_price_delta_dir = f"/mnt/dscc202-datasets/misc/G06/tokenrec/tokenpricetables/"
dbutils.fs.mkdirs(token_price_delta_dir)

# COMMAND ----------

tokenpriceDF.write.format("delta").mode("overwrite").save(token_price_delta_dir)

# COMMAND ----------

spark.sql(
    """
DROP TABLE IF EXISTS g06_db.token_price
"""
)
spark.sql(
    f"""
CREATE TABLE g06_db.token_price
LOCATION "{token_price_delta_dir}"
"""
)

# COMMAND ----------

block_silver=blockDF.withColumn("timestamp", from_unixtime(col("timestamp"),"yyyy-MM-dd HH:mm:ss"))

# COMMAND ----------

dbutils.fs.rm(f'/mnt/dscc202-datasets/misc/G06/tokenrec/blockstables/', recurse=True)
block_delta_dir = f'/mnt/dscc202-datasets/misc/G06/tokenrec/blockstables/'
dbutils.fs.mkdirs(block_delta_dir)

# COMMAND ----------

block_silver.write.format('delta').mode("overwrite").save(block_delta_dir)

# COMMAND ----------

spark.sql(
    """
DROP TABLE IF EXISTS g06_db.block_silver
"""
)
 
spark.sql(
    f"""
CREATE TABLE g06_db.block_silver
USING DELTA
LOCATION "{block_delta_dir}"
"""
)

# COMMAND ----------

erc20_tokentransfer = spark.sql("select token_address,from_address,to_address,value, block_number from g06_db.erc20_token_transfer")

# COMMAND ----------

ddl_schema = StructType([
  StructField("token_address", StringType(), True),
  StructField("from_address", StringType(), True),
  StructField("to_address", StringType(), True),
  StructField("value", DecimalType(38,0), True),   
  StructField("block_number", LongType(), True),
])

assert ddl_schema == _parse_datatype_string("token_address string ,from_address string ,to_address string ,value decimal(38,0), block_number long"), "File not present in Silver Path"

print("Assertion passed.")

# COMMAND ----------

block_date = block_silver.select("number","timestamp")

# COMMAND ----------

token_transfer_selected = erc20_tokentransfer.join(block_date,erc20_tokentransfer.block_number==block_date.number, "left").filter(col("timestamp")>=start_date).select("token_address","from_address","to_address","value")

# COMMAND ----------

ddl_schema = StructType([
  StructField("token_address", StringType(), True),
  StructField("from_address", StringType(), True),
  StructField("to_address", StringType(), True),
  StructField("value", DecimalType(38,0), True),   
])

assert ddl_schema == _parse_datatype_string("token_address string ,from_address string ,to_address string ,value decimal(38,0)"), "File not present in Silver Path"

print("Assertion passed.")

# COMMAND ----------

dbutils.fs.rm(f'/mnt/dscc202-datasets/misc/G06/tokenrec/ttstables/', recurse=True)
token_transfer_selected_delta_dir = f'/mnt/dscc202-datasets/misc/G06/tokenrec/ttstables/'
dbutils.fs.mkdirs(token_transfer_selected_delta_dir)

# COMMAND ----------

token_transfer_selected.write.format("delta").mode("overwrite").save(token_transfer_selected_delta_dir)

# COMMAND ----------

spark.sql(
    """
DROP TABLE IF EXISTS g06_db.token_transfer_selected
"""
)
 
spark.sql(
    f"""
CREATE TABLE g06_db.token_transfer_selected
USING DELTA
LOCATION "{token_transfer_selected_delta_dir}"
"""
)

# COMMAND ----------

token_transfer_selectedDF = spark.sql("select * from G06_db.token_transfer_selected")

# COMMAND ----------

sell=token_transfer_selectedDF.groupBy(['from_address','token_address']).count()

# COMMAND ----------

sell=sell.select("from_address",col('token_address').alias("token_address_sell"),col('count').alias('sell_count'))

# COMMAND ----------

buy=token_transfer_selectedDF.groupBy(['to_address','token_address']).count()

# COMMAND ----------

buy=buy.select("to_address",col('token_address').alias("token_address_buy"),col('count').alias('buy_count'))

# COMMAND ----------

wallet_count=buy.join(sell,(sell.from_address==buy.to_address) & (sell.token_address_sell==buy.token_address_buy))

# COMMAND ----------

wallet_count=wallet_count.select(col('to_address').alias('wallet_address'),col("token_address_buy").alias('token_address'),"buy_count",'sell_count')

# COMMAND ----------

ddl_schema = StructType([
  StructField("wallet_address", StringType(), True),
  StructField("token_address", StringType(), True),  
  StructField("buy_count", LongType(), True),
  StructField("sell_count", LongType(), True),
])

assert ddl_schema == _parse_datatype_string("wallet_address string ,token_address string, buy_count long, sell_count long"), "File not present in Silver Path"

print("Assertion passed.")

# COMMAND ----------

dbutils.fs.rm(f'/mnt/dscc202-datasets/misc/G06/tokenrec/wctables/', recurse=True)
wallet_count_delta_dir = f'/mnt/dscc202-datasets/misc/G06/tokenrec/wctables/'
dbutils.fs.mkdirs(wallet_count_delta_dir)

# COMMAND ----------

wallet_count.write.format('delta').mode("overwrite").save(wallet_count_delta_dir)

# COMMAND ----------

spark.sql(
    """
DROP TABLE IF EXISTS g06_db.wallet_count
"""
)
 
spark.sql(
    f"""
CREATE TABLE g06_db.wallet_count
USING DELTA
LOCATION "{wallet_count_delta_dir}"
"""
)

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
