import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score


class Model_metrics:
    """
        This method calculates model metrics
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def evaluate_model(self, model,x_train, y_train, x_test, y_test):
        print("Train Accuracy :", accuracy_score(y_train, model.predict(x_train)))
        print("Train Confusion Matrix:")
        c_mat_train = confusion_matrix(y_train, model.predict(x_train))
        print(c_mat_train)
        print('Train Recall')
        print(c_mat_train[1, 1] / (c_mat_train[1, 1] + c_mat_train[1, 0]))
        print("-" * 50)
        print("Test Accuracy :", accuracy_score(y_test, model.predict(x_test)))
        print("Test Confusion Matrix:")
        c_mat_test = confusion_matrix(y_test, model.predict(x_test))
        print(c_mat_test)
        print('Test Recall')
        print(c_mat_test[1, 1] / (c_mat_test[1, 1] + c_mat_test[1, 0]))

    def get_accuracy_metrics(self, X_df, Y_df, cut_off, model):
        try:
            X_df['model_probability'] = model.predict_proba(X_df)[:, 1]
            df = X_df.join(Y_df)

            df.reset_index(inplace=True)

            tn, fp, fn, tp, sn, sp, ac, mean_ypred = [], [], [], [], [], [], [], []

            for i in range(len(cut_off)):
                df['Model prediction'] = df['model_probability'].apply(lambda x: 1 if x > cut_off[i] else 0)
                results = confusion_matrix(df['voice_call_quality'], df['Model prediction'])
                mean_ypred.append(round(np.mean(
                    df[df['voice_call_quality'] != 1].loc[df['Model prediction'] == 1, 'model_probability']) * 100, 2))
                ac.append(round(accuracy_score(df['voice_call_quality'], df['Model prediction']) * 100, 2))
                tn.append(results[0][0])
                try:
                    fp.append(results[0][1])
                except:
                    fp.append(0)
                fn.append(results[1][0])
                tp.append(results[1][1])
                sn.append(round(float(tp[i]) / float(tp[i] + fn[i]) * 100, 2))
                sp.append(round(float(tn[i]) / float(tn[i] + fp[i]) * 100, 2))
                fpr, tpr, thresholds = roc_curve(df['voice_call_quality'], X_df['model_probability'])
                roc_auc = auc(fpr, tpr)
                GINI = (2 * roc_auc) - 1

            metrics_df = pd.DataFrame(
                {'probability cut offs': cut_off, 'Accuracy': ac, 'Sensitivity': sn, 'Specificity': sp,
                 'True positive predictions': tp, 'True negative predictions': tn, 'False positive predictions': fp,
                 'False negative predictions': fn, 'mean predicted probability': mean_ypred, 'gini': GINI})
            metrics_df = metrics_df[['gini', 'probability cut offs', 'Accuracy', 'Sensitivity', 'Specificity',
                                     'True positive predictions', 'True negative predictions',
                                     'False positive predictions', 'False negative predictions',
                                     'mean predicted probability']]
            return df, metrics_df
        except Exception as e:
            raise Exception()

