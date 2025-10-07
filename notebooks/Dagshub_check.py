import dagshub
dagshub.init(repo_owner='CodeNeuron58', repo_name='Insightube', mlflow=True)

import mlflow
mlflow.set_experiment('Connection Check')
with mlflow.start_run():
  mlflow.log_param('model', 'logistic regression')
  mlflow.log_metric('Accueacy', 1)
  
  
  # Connection = Passed