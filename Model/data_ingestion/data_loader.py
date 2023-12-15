import pandas as pd
import os
import re

class Data_Getter:
    """
        This class is used to obtaing the data from the source
    """

    def __init__(self, file_object, logger_object):
        self.training_data_path = '../Data/Quality_Data/'
        self.file_object = file_object
        self.logger_object = logger_object
        self.prediction_file = '../Model/Prediction_File/December_MyCall_2022.csv'

    def get_data(self):
        """
                Method Name: get_data
                Description: This method reads the data from source.
                Output: A pandas DataFrame.
                """
        self.logger_object.log(self.file_object, 'Entered the get_data method of the Data_Getter class')
        try:
            dfs = []
            for file_name in os.listdir(self.training_data_path):
                if file_name.endswith('.csv') and '2019' not in file_name:
                    file_path = os.path.join(self.training_data_path, file_name)
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.lower().str.replace(' ', '_')
                    df = df.rename(
                        columns={'calldrop_category': 'call_drop_category', 'inout_travelling': 'in_out_travelling'})

                    df['file_name'] = file_name
                    dfs.append(df)
            self.df_1 = pd.concat(dfs, ignore_index=True)  # reading the data file
            self.logger_object.log(self.file_object,
                                   'Data Load Successful.Exited the get_data method of the Data_Getter class')
            return self.df_1
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_data method of the Data_Getter class. Exception message: ' + str(e))
            self.logger_object.log(self.file_object,'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()

    def rename_file(self, input_filename):
        match = re.search(r'MyCall_Data_([a-zA-Z]+)_(\d{4})\.csv', input_filename)
        if match:
            month, year = match.groups()
            new_filename = f'{month}_{input_filename.split("_")[0]}_{year}.csv'
            return new_filename
        else:
            return input_filename

    def extract_month_year(self, file_name):
        match = re.match(r'(\w+)_(\w+)_(\d{4}).csv', file_name)
        if match:
            month, _, year = match.groups()
            return f"{month}_{year}"
        else:
            return None

    def get_prediction_data(self):
        """
        Method Name: get_prediction_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception


        """
        self.logger_object.log(self.file_object,'Entered the get_data method of the Data_Getter class')
        try:
            self.data= pd.read_csv(self.prediction_file) # reading the data file
            self.data.columns = self.data.columns.str.lower().str.replace(' ', '_')
            self.data = self.data.rename(columns={'calldrop_category': 'call_drop_category', 'inout_travelling': 'in_out_travelling'})

            self.data['file_name'] = 'December_MyCall_2022.csv'
            self.logger_object.log(self.file_object,'Data Load Successful.Exited the get_data method of the Data_Getter class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            self.logger_object.log(self.file_object,
                                   'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()


