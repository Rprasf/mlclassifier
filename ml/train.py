import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import mlflow
from ml.base import BaseEstimator
from ml.dataextraction import DataTraining
import numpy as np
from fastapi import UploadFile

class RandomForestClassifiers(BaseEstimator):

    def __init__(self, datafile: UploadFile, max_features=None, min_samples_split=10, max_depth=None, criterion=None):
        params = {"n_estimators": 4, "max_depth": max_depth , "max_features": max_features}
        self.datafile = datafile
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_for = RandomForestClassifier(**params)


    def split(self, test_size):
        #load data
        data_extractor = DataTraining(self.datafile)
        self.df = data_extractor.load_data()

        #split dataframe
        self.X = self.df.drop(['default'], axis ='columns')
        self.y = self.df['default']
        
        #handle exceptions
        super().train(self.X, self.y)
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(self.X, self.y, test_size = test_size, random_state = 0)
     
    
    def fit(self):
        # Fitting the classifier into the Training set
        model = self.random_for.fit(self.X_Train,self.Y_Train)
        y_predictions = model.predict(self.X_Test)
        #accuracy and graph calculation
        self.metrics(y_predictions)
        return model
        
    def metrics(self, y_predictions):    
        print("Accuracy:", accuracy_score(self.Y_Test, y_predictions))
        print(classification_report(self.Y_Test, y_predictions))

        #calculate confusion matrix
        print(confusion_matrix(self.Y_Test, y_predictions))
        

             
        

