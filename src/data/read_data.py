import os
import json
import os
import json
import pandas as pd
import re

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PROJECT_DIR = os.path.abspath( os.path.join(cwd,'..'))
CODING_DIR = os.path.join(PROJECT_DIR,'data','coding_values') 
GLES_DIR = os.path.join(PROJECT_DIR,'data','raw','GLES')
RAW_DATA_DIR= os.path.join(PROJECT_DIR,'data','raw')

def read_stata_file(file):
    df=pd.read_stata(file, convert_categoricals=False) 
    wave_id = re.search(r'_w(.*?)_', file)
    if wave_id:
        wave_id = wave_id.group(1)
    df['wave_id']=wave_id

    return df

def load_lookup_data(filename):
    with open(os.path.join(filename), 'r') as f:
        data= json.load(f)
        return {int(k): v for k, v in data.items()}

def load_raw_survey_data(wave_number):
    if type(wave_number)== int and (wave_number>=1 and wave_number <= 9):
        wave_fname= 'ZA6838_w1to9_sA_v6-0-0.dta'
        wave_open_ended_fname = f'ZA6838_W{wave_number}_open-ended_v6-0-0.csv'# GLES

        print('wave_fname',wave_fname)
    else:      
        wave_fname = f'ZA6838_w{wave_number}_sA_v6-0-0.dta' #GLES
        wave_open_ended_fname = f'ZA6838_W{wave_number}_sA_open-ended_v6-0-0.csv'# GLES
    
    coding_840s_path = os.path.join(RAW_DATA_DIR,r"ZA7957_6838_v2.0.0.csv") # GESIS-BERT classification for open ended answers 
    wave_df = pd.read_stata(os.path.join(GLES_DIR, wave_fname), convert_categoricals=False)
    wave_open_ended_df = pd.read_csv(os.path.join(GLES_DIR, 'open_ended', wave_open_ended_fname), sep=';', encoding='iso-8859-1')
    df_coding_840s = pd.read_csv(coding_840s_path, sep=';', encoding='iso-8859-1')

    return wave_df, wave_open_ended_df, df_coding_840s

def get_wave_ids():
    wave_ids = [match.group(1) for fname in os.listdir(GLES_DIR) if fname.endswith('.dta') and (match := re.search(r'_w([^_]+)_s', fname))]
    wave_ids = ['a2','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    print(wave_ids)
    return wave_ids

def get_wave_df_dict():
    wave_df_dict={}

    for wave_id in get_wave_ids():
        wave_df, _, _= load_raw_survey_data(wave_id)
        wave_df_dict[wave_id]=wave_df

    return wave_df_dict

if __name__ == "__main__":
    print("PROJECT_DIR",PROJECT_DIR)
    print("CODING_DIR",CODING_DIR)
