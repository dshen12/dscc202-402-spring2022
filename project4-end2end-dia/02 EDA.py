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
# MAGIC ### Rubric for this module
# MAGIC Answer the quetions listed below.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %fs ls /mnt/dscc202-datasets/misc/G06

# COMMAND ----------

# MAGIC %python
# MAGIC # Grab the global variables
# MAGIC wallet_address,start_date = Utils.create_widgets()
# MAGIC print(wallet_address,start_date)
# MAGIC spark.conf.set('wallet.address',wallet_address)
# MAGIC spark.conf.set('start.date',start_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What is the maximum block number and date of block in the database

# COMMAND ----------

# MAGIC %sql
# MAGIC select number as MAX_BLOCK_NUMBER,
# MAGIC from_unixtime(timestamp, "yyyy/MM/dd HH:MM:SS") as DATE_OF_MAX_BLOCK_NUMBER
# MAGIC from ethereumetl.blocks
# MAGIC where number in (select max(number)
# MAGIC                  from ethereumetl.blocks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: At what block did the first ERC20 transfer happen?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT number as BLOCKNUMBER_FIRST_ERC20
# MAGIC FROM (SELECT number, timestamp
# MAGIC FROM ethereumetl.blocks 
# MAGIC INNER JOIN ethereumetl.token_transfers
# MAGIC ON ethereumetl.blocks.number = ethereumetl.token_transfers.block_number 
# MAGIC ORDER BY timestamp ASC
# MAGIC LIMIT 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: How many ERC20 compatible contracts are there on the blockchain?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT COUNT(is_erc20) as Number_Of_ERC20
# MAGIC FROM ethereumetl.silver_contracts
# MAGIC WHERE is_erc20 = "True"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Q: What percentage of transactions are calls to contracts

# COMMAND ----------

# MAGIC %python 
# MAGIC transactions_to_adress_DF = spark.sql('SELECT COUNT (to_address) FROM ethereumetl.transactions INNER JOIN ethereumetl.silver_contracts ON ethereumetl.transactions.to_address = ethereumetl.silver_contracts.address')

# COMMAND ----------

# MAGIC %python
# MAGIC transactions_to_adress_all_DF = spark.sql('SELECT COUNT (to_address) FROM ethereumetl.transactions')

# COMMAND ----------

# MAGIC %python
# MAGIC n1 = transactions_to_adress_DF.first()['count(to_address)']
# MAGIC n2 = transactions_to_adress_all_DF.first()['count(to_address)']

# COMMAND ----------

from pyspark.sql.functions import round 
perc =  1 - n1/n2
print(perc*100,'% of transactions are calls to contracts')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What are the top 100 tokens based on transfer count?

# COMMAND ----------

# MAGIC %sql
# MAGIC select token_address,count(transaction_hash) tran_cnt
# MAGIC from ethereumetl.token_transfers
# MAGIC group by token_address
# MAGIC order by tran_cnt desc
# MAGIC limit 100

# COMMAND ----------

# Python Approach

# COMMAND ----------

transferDF = spark.sql('select * from ethereumetl.token_transfers')

# COMMAND ----------

from pyspark.sql.functions import col, count, mean,first

transferDF = transferDF.groupBy("token_address").agg(sum("value").alias("transfer_count_for_each_token_address"))
q5 = transferDF.sort(col("transfer_count_for_each_token_address").desc()).limit(100)

# COMMAND ----------

token_join = spark.sql('select address,symbol,name from ethereumetl.tokens')
new = token_join.join(q5, token_join.address == q5.token_address,"inner").select(q5.token_address, token_join.symbol, q5.transfer_count_for_each_token_address) 
new = new.groupBy("token_address").agg(mean("transfer_count_for_each_token_address").alias("transfer_count_for_each_symbol"), first("symbol").alias("symbol"))
new = new.sort(col("transfer_count_for_each_symbol").desc())

# COMMAND ----------

display(new)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What fraction of ERC-20 transfers are sent to new addresses
# MAGIC (i.e. addresses that have a transfer count of 1 meaning there are no other transfers to this address for this token this is the first)

# COMMAND ----------

# MAGIC %sql
# MAGIC select a.erc20_new_cnt/b.erc20_cnt
# MAGIC from 
# MAGIC (
# MAGIC   select count(*) erc20_new_cnt
# MAGIC   from
# MAGIC   (
# MAGIC     select token_address,count(transaction_hash) tran_cnt
# MAGIC     from ethereumetl.token_transfers
# MAGIC     group by token_address
# MAGIC     having tran_cnt==1
# MAGIC   )
# MAGIC )a
# MAGIC join
# MAGIC (
# MAGIC   select count(distinct token_address) erc20_cnt
# MAGIC   from ethereumetl.token_transfers
# MAGIC )b

# COMMAND ----------

# PYTHON
transferDF1 = spark.sql('select token_address, from_address, to_address, transaction_hash from ethereumetl.token_transfers')

# COMMAND ----------

# take a look at the data with 100 rows selected

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT token_address, from_address, to_address, transaction_hash, block_number
# MAGIC FROM ethereumetl.token_transfers
# MAGIC LIMIT 100

# COMMAND ----------

# group by certain token address and the corresponding to_address and count the transfer
ERC20_TRANSFER = transferDF1.groupBy("token_address","to_address").agg(count("transaction_hash").alias("transfer_count"))
# extract those with transfer count of 1 
ERC20_TRANSFER_new = ERC20_TRANSFER.where(ERC20_TRANSFER.transfer_count == 1)

# COMMAND ----------

new_count = ERC20_TRANSFER_new.count()
overall_count = ERC20_TRANSFER.count()

# COMMAND ----------

percents = (new_count/overall_count)*100

# COMMAND ----------

print(percents, '% of ERC-20 transfers are sent to new addresses')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: In what order are transactions included in a block in relation to their gas price?
# MAGIC - hint: find a block with multiple transactions 

# COMMAND ----------

transactionsDF = spark.sql('select hash, block_number, transaction_index, gas, gas_price from ethereumetl.transactions')
# sample #48315 block
sample = transactionDF.where(transactionDF.block_number = 48315)
sample = sample.sort(sample.transaction_index)
sample.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### - From the above table, we can find that transactions with higher gas price will be included in a block earlier. In other words, transaction_index integer for the transactions executed in the same block will be larger for those who has lower gas price

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What was the highest transaction throughput in transactions per second?
# MAGIC hint: assume 15 second block time

# COMMAND ----------

transactionDF = spark.sql('select block_number, hash from ethereumetl.transactions')
transactionDF = transactionDF.groupBy("block_number").agg(count('hash').alias("Transactions_Count"))
transactionDF = transactionDF.withColumn("throughput", (transactionDF.Transactions_Count)/15)
transactionDF = transactionDF.sort(col('throughput').desc())
transactionDF = transactionDF.select("throughput").limit(1)

# COMMAND ----------

display(transactionDF)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT (max(transaction_count) / 15) AS highest_transaction_throughput
# MAGIC FROM ethereumetl.blocks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What is the total Ether volume?
# MAGIC Note: 1x10^18 wei to 1 eth and value in the transaction table is in wei

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT sum(value)/(power(10,18))
# MAGIC AS total_Ether_volume
# MAGIC FROM ethereumetl.transactions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: What is the total gas used in all transactions?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT SUM(gas_used)
# MAGIC AS total_gas_used
# MAGIC FROM ethereumetl.receipts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: Maximum ERC-20 transfers in a single transaction

# COMMAND ----------

# MAGIC %sql
# MAGIC select  *
# MAGIC from ethereumetl.token_transfers
# MAGIC order by value desc
# MAGIC limit 1

# COMMAND ----------

ERC20_TRANSFER = spark.sql('SELECT * FROM ethereumetl.token_transfers')
ERC20_TRANSFER1 = ERC20_TRANSFER.sort(col("value").desc())
# show the row with the maximum amount of transfers of ERC-20
ERC20_TRANSFER1.limit(1).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q: Token balance for any address on any date?

# COMMAND ----------

# wallet_address = '0xff29d3e552155180809ea3a877408a4620058086'
# date =  2022-01-07 09:01:28

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

# COMMAND ----------

erc20_tokentransfer = spark.sql("select token_address,from_address,to_address,value, block_number from g06_db.erc20_token_transfer")

erc20_blocks = spark.sql("select number, timestamp from g06_db.block_date_silver")

erc20_tokentransfer = erc20_tokentransfer.join(erc20_blocks, erc20_tokentransfer.block_number == erc20_blocks.number, "inner").select("token_address", "from_address", "to_address", "value", "block_number", "timestamp")

# COMMAND ----------

fromDF = erc20_tokentransfer.groupBy("timestamp","from_address").agg(sum(erc20_tokentransfer.value).alias("send_value")).withColumn("address",col("from_address"))

# COMMAND ----------

toDF = erc20_tokentransfer.groupBy("timestamp","to_address").agg(sum(erc20_tokentransfer.value).alias("receive_value")).withColumn("address",col("to_address"))

# COMMAND ----------

balance = fromDF.join(toDF, ["address", "timestamp"], "inner").withColumn("balance", col("receive_value") - col("send_value"))

# COMMAND ----------

balance = balance.select("address","timestamp","balance")
balancechoice = balance.where((col("address") == "0xff29d3e552155180809ea3a877408a4620058086"))
balancechoice = balancechoice.where(col("timestamp") == "2022-01-07 09:01:28")

# COMMAND ----------

display(balancechoice)

# COMMAND ----------

# sql approach
# wallet_address = '0xff29d3e552155180809ea3a877408a4620058086'
# date =  2022-01-07 09:01:28

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT address, sum(value) AS balance, timestamp
# MAGIC FROM 
# MAGIC 
# MAGIC (SELECT value,block_number, timestamp, to_address AS address
# MAGIC FROM g06_db.erc20_token_transfer
# MAGIC INNER JOIN g06_db.block_date_silver
# MAGIC ON g06_db.erc20_token_transfer.block_number = g06_db.block_date_silver.number
# MAGIC UNION
# MAGIC SELECT -value, block_number, timestamp, from_address AS address
# MAGIC FROM g06_db.erc20_token_transfer
# MAGIC INNER JOIN g06_db.block_date_silver
# MAGIC ON g06_db.erc20_token_transfer.block_number = g06_db.block_date_silver.number)
# MAGIC 
# MAGIC WHERE address = '0xff29d3e552155180809ea3a877408a4620058086'
# MAGIC AND timestamp = "2022-01-07 09:01:28"
# MAGIC GROUP BY address, timestamp

# COMMAND ----------

# MAGIC %md
# MAGIC ## Viz the transaction count over time (network use)

# COMMAND ----------

from pyspark.sql.functions import col, count, mean,first
trans_block = spark.sql('select number, from_unixtime(timestamp, "yyyy/MM/dd") AS Date from ethereumetl.blocks')
trans = spark.sql('select hash, block_number from ethereumetl.transactions')
trans = trans.groupBy('block_number').agg(count("hash").alias("number_of_transactions_in_this_block"))
trans_overtime = trans_block.join(trans, trans_block.number == trans.block_number, "inner")
trans_overtime = trans_overtime.select('number_of_transactions_in_this_block','block_number','Date')
trans_overtime = trans_overtime.groupBy("Date").agg(sum("number_of_transactions_in_this_block").alias("transactions_per_day"))

# COMMAND ----------

display(trans_overtime)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Viz ERC-20 transfer count over time
# MAGIC interesting note: https://blog.ins.world/insp-ins-promo-token-mixup-clarified-d67ef20876a3

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
trans = spark.sql('select number, from_unixtime(timestamp, "yyyy/MM/dd") AS Date from ethereumetl.blocks')
token_block = spark.sql('select value, block_number from ethereumetl.token_transfers')
ERC20_overtime = trans.join(token_block, trans.number == token_block.block_number,'inner')
ERC20_overtime = ERC20_overtime.select('value','block_number','Date')
ERC20_overtime = ERC20_overtime.groupBy("Date").agg(sum("value").alias("ERC20_Transfer_Amount"))


# COMMAND ----------

display(ERC20_overtime)

# COMMAND ----------

# MAGIC %python
# MAGIC # Return Success
# MAGIC dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
