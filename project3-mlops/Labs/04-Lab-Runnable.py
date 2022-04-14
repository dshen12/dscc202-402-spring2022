# Databricks notebook source
# TODO
# Create 3 widgets for parameter passing into the notebook:
#   - n_estimators with a default of 100
#   - learning_rate with a default of .1
#   - max_depth with a default of 1 
# Note that only strings can be used for widgets
dbutils.widgets.text("n_estimators","100")
dbutils.widgets.text("learning_rate","0.1")
dbutils.widgets.text("max_depth","1")

# COMMAND ----------

# TODO
#Read from the widgets to create 3 variables.  Be sure to cast the values to numeric types
n_estimators = int(dbutils.widgets.get("n_estimators"))
learning_rate = float(dbutils.widgets.get("learning_rate"))
max_depth = int(dbutils.widgets.get("max_depth"))

# COMMAND ----------

# TODO
# Train and log the results from a model.  Try using Gradient Boosted Trees
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
import mlflow.sklearn
with mlflow.start_run() as run:

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    
    gbr = GradientBoostingRegressor(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth)
    
    mlflow.sklearn.log_model(gbr, "Gradient-Boosting-Regressor")
  
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
  
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
    artifactURI=mlflow.get_artifact_uri()
  
    print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

# TODO
#Report the model output path to the parent notebook

import json
 
dbutils.notebook.exit(json.dumps({
  "status": "OK",
  #"data_input_path": data_input_path,
  "run_id":runID,
  #"path":path,
  #"data_output_path": full_path.replace("dbfs:", "/dbfs")
}))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
