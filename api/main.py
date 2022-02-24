from fastapi import FastAPI,UploadFile
from fastapi import BackgroundTasks
import mlflow 
import pandas as pd
from random import random, randint
import json
import os
from fastapi.responses import StreamingResponse
from fastapi import Query
import io
import uuid
import time 

from mlflow.tracking import MlflowClient
from ml.train import RandomForestClassifiers

app = FastAPI(title='Train/Predict ML Models API')
mlflow.set_tracking_uri("sqlite:///db/backend.db")
mlflowclient = MlflowClient(mlflow.get_tracking_uri(), mlflow.get_registry_uri())


"FAST API for training model"
"start mlflow :"
"load data (call data cleaning, dataingestion, etc) "
"1. split dataset :"
"2. call 'RandomForest' from Factory "
"3. fit and accuracy cal"
"4.log params and metric in ml flow"
"5.register and save model in mlflow"
"6.finish ml flow experiment"
"7. push latest version of registered model and push to production"


def train_model_task(model_name: str = Query(None, description="type model_name to register"), trainFile: UploadFile  = Query(None, description="input file to train model")):
    "Todo : modify to a heavy task runner like Celery"
    try:
        #MLflow tracking
        mlflow.set_experiment('RandomForestClassifierExp')
        with mlflow.start_run() as run:
            # Log parameters and metrics using the MLflow APIs (generally hyperparameters)
            mlflow.log_param("param_1", randint(0, 100))
            mlflow.log_metrics({"metric_1": random(), "metric_2": random() + 1})

            #load model according to model_name, at present loads RFC by default. can be extended
            model_instance = RandomForestClassifiers(trainFile)
            
            # Load data and model
            model_instance.split(0.25)
            
            classifier_model =  model_instance.fit()
            
            # Register model 
            mlflow.sklearn.log_model(sk_model=classifier_model,artifact_path="",registered_model_name=model_name)
        
            # Transition to production. We search for the last model with the name and we stage it to production
            mv = mlflowclient.search_model_versions("name='{}'".format(model_name))[-1] 
            
            # Take last model version
            mlflowclient.transition_model_version_stage( name=mv.name, version=mv.version, stage="production")   
         
    except Exception as e:
        print('Error in model training')
        print('ExceptionStackTrace: {0}'.format(e))
        raise e


@app.post("/train")
async def train_api(model_name : str, trainFile: UploadFile, background_tasks: BackgroundTasks):
   background_tasks.add_task(train_model_task, model_name, trainFile)
   id = uuid.uuid4()
   startTime = time.time()
   return {'id': id,'result':'training started', 'startTime': startTime}


"predict (data_inference, modelname)"
"1. ml load model"
"2. predict model"
"3. analyse any metrics for accuracy"
"4. return result"

@app.post('/predict')
async def predict_api(model_name : str,  customerFile: UploadFile):
    #Pick model from registry
    model = mlflow.sklearn.load_model(
        model_uri=f"models:/{model_name}/Production"
    )
    #load data
    data = pd.read_csv(customerFile.file)
    #Predict
    res = model.predict(data)
    data['default'] = res.tolist()
    #transform dataframe to csv file column
    stream = io.StringIO()
    data.to_csv(stream, index = False)
    response = StreamingResponse(iter([stream.getvalue()]),
                            media_type="text/csv")
   
    # tests model with hardcoded values
    #print("test:",model.predict([[9803,77,100,1,0,3,17.237973957712,8,76.3751272014125]]))
    
    response.headers["Content-Disposition"] = "attachment; filename=loan-default.csv"
    return response
    

"API : to load model from ml flow"
"1.get models using mlflowclient"

@app.get('/models')
async def get_models():
    # Read models from mlflow registry
    models = mlflowclient.list_registered_models()
    models = [model.name for model in models]
    return models