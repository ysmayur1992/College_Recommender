import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict1(self,features):
        try:
            model_path=os.path.join("artifacts","model1.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor1.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict2(self,features):
        try:
            model_path=os.path.join("artifacts","model2.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor2.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData1:
    def __init__(  self,
        Maths: int,
        Science: int,
        History: int,
        Geography: int,
        Language: int,
        Gender: str,
        Locality: str,
        Budget: int
        ):

        self.Maths = Maths
        self.Science = Science
        self.History = History
        self.Geography = Geography
        self.Language = Language
        self.Gender = Gender
        self.Locality = Locality
        self.Budget = Budget


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Maths": [self.Maths],
                "General Science": [self.Science],
                "History": [self.History],
                "Geography": [self.Geography],
                "Language": [self.Language],
                "Gender":[self.Gender],
                "Locality":[self.Locality],
                "Budget":[self.Budget]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        


class CustomData2:
    def __init__(  self,
        Physics: int,
        Maths: int,
        Chemistry: int,
        Biology: int,
        Entrance: int,
        Locality: str,
        prev_stream: str,
        Budget: int
        ):

        self.Physics = Physics
        self.Maths = Maths
        self.Chemistry = Chemistry
        self.Biology = Biology
        self.Entrance = Entrance
        self.Locality = Locality
        self.prev_stream = prev_stream
        self.Budget = Budget


    def get_data_as_data_frame(self):
        try:
            data_frame = {
                "Physics": [self.Physics],
                "Maths": [self.Maths],
                "Chemistry": [self.Chemistry],
                "Biology": [self.Biology],
                "Locality":[self.Locality],
                "Budget":[self.Budget],
                "Entrance": [self.Entrance],
                "prev_stream":[self.prev_stream]
            }

            return pd.DataFrame(data_frame)

        except Exception as e:
            raise CustomException(e, sys)