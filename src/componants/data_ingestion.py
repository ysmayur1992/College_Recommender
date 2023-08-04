import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.componants.data_transformation import Data_Transformation
from src.componants.data_transformation import Data_Transformation_Config

from src.componants.model_trainer import ModelTrainerConfig
from src.componants.model_trainer import ModelTrainer
@dataclass
class Data_Ingestion_Config:
    raw_data_path1: str=os.path.join('artifacts',"data_for_10th.csv")
    train_data_path1: str=os.path.join('artifacts',"train_10th.csv")
    test_data_path1: str=os.path.join('artifacts',"test_10th.csv")

    raw_data_path2: str=os.path.join('artifacts',"data_for_12th.csv")
    train_data_path2: str=os.path.join('artifacts',"train_12th.csv")
    test_data_path2: str=os.path.join('artifacts',"test_12th.csv")

class Data_Ingestion:
    def __init__(self):
        self.ingestion_config=Data_Ingestion_Config()

    def initiate_data_ingestion_for_10th(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df_10th=pd.read_csv('Data\completed_10th.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path1),exist_ok=True)

            df_10th.to_csv(self.ingestion_config.raw_data_path1,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df_10th,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path1,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path1,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path1,
                self.ingestion_config.test_data_path1
            )
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_ingestion_for_12th(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df_12th=pd.read_csv('Data\completed_12th.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path1),exist_ok=True)

            df_12th.to_csv(self.ingestion_config.raw_data_path2,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df_12th,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path2,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path2,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path2,
                self.ingestion_config.test_data_path2
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=Data_Ingestion()
    train_path1,test_path1=obj.initiate_data_ingestion_for_10th()
    train_path2,test_path2 = obj.initiate_data_ingestion_for_12th()

    data_transformation=Data_Transformation()
    train_set1,test_set1,_=data_transformation.initiate_data_transformation_for_10th(train_path1,test_path1)
    train_set2,test_set2,_ = data_transformation.initiate_data_transformation_for_12th(train_path2,test_path2)

    modeltrainer=ModelTrainer()
    name,score = modeltrainer.initiate_model_training_for_10th(train_array=train_set1,test_array=test_set1)
    message = "The best model is {0} and it's score is {1}".format(name,score)
    print(message)

    name,score = modeltrainer.initiate_model_training_for_12th(train_array=train_set2,test_array=test_set2)
    message = "The best model is {0} and it's score is {1}".format(name,score)
    print(message)