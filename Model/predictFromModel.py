import pickle
import os
import pandas
from data_preprocessing import preprocessing
from data_ingestion import data_loader
from application_logging import logger




class prediction:

    def __init__(self):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()

    def predictionFromModel(self):

        try:
            print('prediction started')
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter = data_loader.Data_Getter(self.file_object, self.log_writer)
            data = data_getter.get_prediction_data()

            print('prediction data read')

            raw_data = data.copy()

            data['file_name'] = data['file_name'].apply(data_getter.rename_file)

            data['month_year'] = data['file_name'].apply(data_getter.extract_month_year)

            data['year'] = data['month_year'].apply(lambda x: x.split('_')[1])

            data['month'] = data['month_year'].apply(lambda x: x.split('_')[0])

            print('preprocessor started')

            """doing the data preprocessing"""

            data['network_type'] = data['network_type'].fillna('Unknown')

            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
            cols_to_remove = ['rating', 'state_name', 'file_name', 'month_year', 'year']
            data = preprocessor.remove_columns(data, cols_to_remove)

            print('preprocessing done')

            # As there are more number of operator unique values we will restrict it to 5 unique values whose frequency is more and make others as other.
            op_levels = data['operator'].value_counts()[:5].index.tolist()
            data['operator'] = data['operator'].apply(lambda x: x if x in op_levels else 'Other')

            # Dummy df columns for categorical variables
            cols_to_dummy = ['operator', 'in_out_travelling', 'network_type', 'month']
            data = preprocessor.make_dummy_df(data, cols_to_dummy=cols_to_dummy)

            print('dummy cols done')

            ## Load the model
            product_path = 'Training_metrics/'
            output_fname = "final_model.pkl"
            try:
                with open(product_path + output_fname, 'rb') as file:
                    final_model, model_features = pickle.load(file)
            except Exception as e:
                print('error loading model')
                print(e)
            print('model loaded')
            missing_col = list(set(model_features) - set(list(data.columns)))
            extra_col = list(set(list(data.columns)) - set(model_features))
            data.drop(extra_col, axis=1)
            for col in missing_col:
                data[col] = 0
            data = data[model_features]

            raw_data['model_probability'] = final_model.predict_proba(data.values)[:, 1]

            raw_data['model_prediction'] = raw_data['model_probability'].apply(lambda x : 'Poor Voice Quality' if x > 0.29 else 'Satisfactory')

            # raw_data['call_drop_category_predicted'] = raw_data['model_prediction'].map({0 : 'Satisfactory', 1 : 'Poor Voice quality'})
            print('prediction data prepared')

            raw_data.to_csv("Prediction_Output_File/Predictions.csv",header=True,mode='a+') #appends result to prediction file
            print('prediction file saved')
            self.log_writer.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise Exception
        return raw_data.head().to_dict(orient='records')




