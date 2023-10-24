import pandas as pd
import numpy as np
import sdv
from sdv.evaluation.single_table import evaluate_quality

from ML_efficiency import MLE
from discriminator import Discriminator


class Evaluation():

    def __init__(self, synthetic_datasets, synthetic_datasets_names,
               real_dataset, target_column ='GoodCustomer'):
        
        #synthetic_datasets must be a list
        #synthetic_datasets_names must be a list of the names of the datasets (to create the comparison)
        
        self.synthetic_datasets = synthetic_datasets
        self.synthetic_datasets_names = synthetic_datasets_names
        self.real_dataset = real_dataset
        self.target_column = target_column

        self.complete_evaluation = pd.DataFrame()
        

    def eval_MLE(self, dataset, dataset_name):
        #synthetic_datasets must be a list
        #synthetic_datasets_names must be a list of the names of the datasets (to create the comparison)
        #print(dataset.columns)

        mle = MLE(synthetic_dataframe=dataset,
            real_dataframe = self.real_dataset,
            target_column = self.target_column)

        mle.preprocessing()

        synthetic_acc, real_acc = mle.eval()

        results = {'dataset': dataset_name,
                    'MLE_syn_1':   synthetic_acc[0],
                    'MLE_real_1' :  real_acc[0],
                    'MLE_syn_2'  :   synthetic_acc[1],
                    'MLE_real_2' :  real_acc[1],
                    'MLE_syn_3'  :   synthetic_acc[2],
                    'MLE_real_3' :  real_acc[2],
                    'MLE_syn_4'  :   synthetic_acc[3],
                    'MLE_real_4' :  real_acc[3]
        }
        
        return results

    def eval_Discriminator(self, dataset, dataset_name):
        discriminator  = Discriminator(synthetic_dataframe=dataset,
                               real_dataframe=self.real_dataset)

        discriminator.preprocessing()

        acc_discriminator = discriminator.fit()
        results = {'dataset': dataset_name,
                   'acc_discriminator': acc_discriminator
        }

        return results
    
    def eval_Sdv(self, dataset, dataset_name):

        real_duplicates = len(self.real_dataset)-len(self.real_dataset.drop_duplicates())
        synthetic_duplicates = len(self.real_dataset.append(dataset))-len(self.real_dataset.append(dataset).drop_duplicates())

        new_duplicates = synthetic_duplicates - real_duplicates

        metadata = sdv.metadata.SingleTableMetadata()
        metadata.detect_from_dataframe(data = self.real_dataset)

        quality_report = evaluate_quality(
                        self.real_dataset,
                        dataset,
                        metadata
                    )
        
        score = quality_report.get_score()


        results = {'dataset': dataset_name,
                   'duplicates': new_duplicates,
                   'sdv_score': score
        }

        return results
         


    

    def eval(self):

        for i, dataset in enumerate(self.synthetic_datasets):
            name = self.synthetic_datasets_names[i]

            results = {}

            results_sdv = self.eval_Sdv(dataset = dataset,
                                        dataset_name=name) 
            results.update(results_sdv)

            results_mle = self.eval_MLE(dataset = dataset,
                                        dataset_name=name)            
            results.update(results_mle)

            results_discriminator = self.eval_Discriminator(dataset = dataset,
                                        dataset_name=name)            
            results.update(results_discriminator)

            if i == 0:
                self.complete_evaluation = pd.DataFrame(results, index=[0])
            
            else:
                self.complete_evaluation = pd.concat([self.complete_evaluation, pd.DataFrame(results, index=[0])], axis = 0)





        



    

