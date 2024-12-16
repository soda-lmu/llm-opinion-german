import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os 
from itertools import islice
from src.bert.bert_classifier import BertClassifier
from concurrent.futures import ThreadPoolExecutor
import json

def classify_experiment_folder(model:BertClassifier,folder_path):
    json_files = os.listdir(folder_path)

    for json_file in tqdm(json_files):
        full_path = os.path.join(folder_path, json_file)

        with open(full_path, 'r') as file:
            data = json.load(file)
        
        if f'classification_{model.class_mode}' in data and data[f'classification_{model.class_mode}']['model_name'] == model.model_name:
                #logger.info(f'classification_{self.class_mode} was done for this file with this model, skipping {json_file}')
                pass

        result = model.predict_text(data['output'])
        if isinstance(result, np.floating):
            result = result.item()
        data[f'classification_{model.class_mode}'] = {'model_name': model.model_name, **result}

        with open(full_path, 'w') as file:
            #logger.info(f'classified {json_file}')
            json.dump(data, file)



def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    
def get_experiment_df(directory):
    all_files = os.listdir(directory)
    json_files = [file for file in all_files if file.endswith('.json')]
    full_paths = [os.path.join(directory, file) for file in json_files]

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(load_json, full_paths), total=len(full_paths)))

    df = pd.DataFrame(results)
    df = df.drop(['runtime', 'prompt', 'input_num_token', 'output_num_token', 'temperature', 'do_sample', 'max_new_tokens'], axis=1, errors='ignore')

    return df

# def get_experiment_df_old(directory):
#     all_files = os.listdir(directory)
#     json_files = [file for file in all_files if file.endswith('.json')]
#     full_paths = [os.path.join(directory, file) for file in json_files]

#     results = []
#     for filepath in tqdm(full_paths):
#         with open(filepath, "r", encoding="utf-8") as f:
#             data = json.load(f)
#             results.append(data)
    
#     df = pd.DataFrame(results)
#     df = df.drop(['runtime', 'prompt', 'input_num_token', 'output_num_token', 'temperature', 'do_sample', 'max_new_tokens'], axis=1, errors='ignore')

#     return df


def get_pred_matrix(self,df):
    pred_matrix = pd.DataFrame(df['predictions'].to_list(), columns=[self.label2str[i] for i in range(0,self.num_classes ) ] )
    return pred_matrix
