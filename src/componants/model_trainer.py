import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path1=os.path.join("artifacts","model1.pkl")
    trained_model_file_path2=os.path.join("artifacts","model2.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_training_for_10th(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            classifier_models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "CatBoosting Regressor": CatBoostClassifier(verbose=False),
                "AdaBoost Regressor": AdaBoostClassifier(),
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=classifier_models)
            
            sorted_model = dict(sorted(model_report.items(),key= lambda x:x[1],reverse=True))

            best_model_name = list(sorted_model.keys())[0]
            best_model = classifier_models[best_model_name]
            best_model_score = list(sorted_model.values())[0]
            
            #if model_report[best_model_name] < 0.6:
                #raise CustomException("No good model found... all the models have low accuracy")            
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path1,
                obj=best_model
            )

            
            return best_model_name,best_model_score
            
            
        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_model_training_for_12th(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            classifier_models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "CatBoosting Regressor": CatBoostClassifier(verbose=False),
                "AdaBoost Regressor": AdaBoostClassifier(),
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=classifier_models)
            
            sorted_model = dict(sorted(model_report.items(),key= lambda x:x[1],reverse=True))

            best_model_name = list(sorted_model.keys())[0]
            best_model = classifier_models[best_model_name]
            best_model_score = list(sorted_model.values())[0]
            
            #if model_report[best_model_name] < 0.6:
                #raise CustomException("No good model found... all the models have low accuracy")            
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path2,
                obj=best_model
            )

            return best_model_name,best_model_score
            
            
        except Exception as e:
            raise CustomException(e,sys)