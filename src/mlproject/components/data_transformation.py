import sys
from dataclasses import dataclass


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.utils import save_object

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import os


#import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MailPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Nothing to learn

    def transform(self, X):
        # Replace NaNs with empty strings
        X = X.where(pd.notnull(X), '')
        
        # Convert 'spam' to 0, 'ham' to 1
        X = X.copy()  # Avoid changing original dataframe
        if 'Category' in X.columns:
            X['Category'] = X['Category'].map({'spam': 0, 'ham': 1})
        return X



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

# def Remove_nulls(raw_mail_data):
#     raw_mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# def Mapping(mail_data):
#     # label spam mail as 0;  ham mail as 1;
#     mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
#     mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            categorical_columns = ["Message"]

            cat_pipeline = Pipeline(steps=[
                ('preprocessor', MailPreprocessor())
            ])

            logging.info(f"Categorical Columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transormation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Category"
            # numerical_columns = ["writing_score", "reading_score"]

            ## divide the train dataset to independent and dependent feature

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            ## divide the test dataset to independent and dependent feature

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )








        except Exception as e:
            raise CustomException(sys,e)
        
if __name__ == "__main__":
    obj = DataTransformation()
    train_path = "/home/shaktil/Practice/artifacts/train.csv"
    test_path = "/home/shaktil/Practice/artifacts/test.csv"
    obj.initiate_data_transormation(train_path, test_path)