import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from pathlib import Path
import numpy as np
import joblib
from pyhelpers.store import save_json
from MLopsProject.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/Axelblaze1O1/MLOPs_Project.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"]="Axelblaze1O1"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="2f299bdb358df25fc3e9d895629e3c7838ab70f8"
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column],axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_tracking_uri("https://dagshub.com/Axelblaze1O1/MLOPs_Project.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

  
        # print(mlflow.get_tracking_uri())
        # print(tracking_url_type_store)
    

        with mlflow.start_run():
     
            predicted_qualities = model.predict(test_x)
            (rmse, mae , r2) = self.eval_metrics(test_y, predicted_qualities)

            scores = {"rmse":rmse,"mae":mae,"r2":r2}
            save_json(path_to_file=Path(self.config.metric_file_name),data=scores)
        
            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse",rmse)
            mlflow.log_metric("r2",r2)
     
            mlflow.log_metric("mae",mae)
      

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model,"model",registered_model_name="ElasticnetModel")
          
            else:
                mlflow.sklearn.log_model(model,"model")
        
