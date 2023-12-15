"""
This is the Entry point for Training the Machine Learning Model.

"""

# Doing the necessary imports
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from best_model_finder import tuner, model_metrics
from application_logging import logger

#Creating the common Logging object


class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()

            data['file_name'] = data['file_name'].apply(data_getter.rename_file)

            data['month_year'] = data['file_name'].apply(data_getter.extract_month_year)

            data['year'] = data['month_year'].apply(lambda x: x.split('_')[1])

            data['month'] = data['month_year'].apply(lambda x: x.split('_')[0])


            """doing the data preprocessing"""

            data['network_type'] = data['network_type'].fillna('Unknown')

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            cols_to_remove = ['rating', 'state_name', 'file_name', 'month_year', 'year']
            data=preprocessor.remove_columns(data,cols_to_remove) # remove the unnamed column as it doesn't contribute to prediction.

            # As there are more number of operator unique values we will restrict it to 5 unique values whose frequency is more and make others as other.
            op_levels = data['operator'].value_counts()[:5].index.tolist()
            data['operator'] = data['operator'].apply(lambda x: x if x in op_levels else 'Other')

            # Dummy df columns for categorical variables
            cols_to_dummy = ['operator', 'in_out_travelling', 'network_type', 'month']
            data = preprocessor.make_dummy_df(data, cols_to_dummy=cols_to_dummy)

            # Target variable creation
            data['voice_call_quality'] = data['call_drop_category'].map(lambda x: 0 if x == 'Satisfactory' else 1)

            data = preprocessor.remove_columns(data, ['call_drop_category'])


            # create separate features and labels
            X,Y=preprocessor.separate_label_feature(data,label_column_name='voice_call_quality')

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=355)

            model_finder=tuner.Model_Finder(self.file_object,self.log_writer) # object initialization

            best_model = model_finder.get_best_params_for_random_forest(x_train,y_train)

            # model evaluation

            model_metric_obj = model_metrics.Model_metrics(self.file_object, self.log_writer)

            model_metric_obj.evaluate_model(best_model,x_train,y_train,x_test,y_test)

            print('model evaluate completed')
            cut_off = np.linspace(0, 1, 25)
            try:
                train_df, train_metrics_df = model_metric_obj.get_accuracy_metrics(x_train,y_train,cut_off,best_model)
            except Exception as e:
                print('error in get accuracy metrics')
                print(e)


            test_df, test_metrics_df = model_metric_obj.get_accuracy_metrics(x_test, y_test, cut_off, best_model)

            print('mdoel evaluation completed')
            try:
                train_metrics_df.to_excel('Training_metrics/train_metrics.xlsx')
            except Exception as e:
                print(e, 'Error in saving excel')
            test_metrics_df.to_excel('Training_metrics/test_metrics.xlsx')

            print('metrics file saved')

            model_to_save = (best_model, X.columns.tolist())

            pickle.dump(model_to_save, open('Training_metrics/final_model.pkl', 'wb'))

            print('model saved')

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception