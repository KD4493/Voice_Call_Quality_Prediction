import pandas as pd
import numpy as np


class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.
        """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self,data,columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception

        """
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data=data
        self.columns=columns
        try:
            self.useful_data=self.data.drop(labels=self.columns, axis=1) # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def make_dummy_df(self, data, cols_to_dummy):
        """
                        Method Name: make_dummy_df
                        Description: This method makes dummy columns for categorical features.
                        Output: Return dataframe which includes dummy columns .
                        On Failure: Raise Exception

                """
        self.logger_object.log(self.file_object, 'Entered the make_dummy_df method of the Preprocessor class')
        try:
            dummy_df = pd.get_dummies(data[cols_to_dummy])
            data = pd.concat([data, dummy_df],axis=1)
            data = data.drop(columns=cols_to_dummy)
            self.logger_object.log(self.file_object,
                                   'Dummy df method successful. Exited the make_dummy_df method of the Preprocessor class')
            return data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in make_dummy_df method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Dummy df Unsuccessful. Exited the make_dummy_df method of the Preprocessor class')
            raise Exception()




