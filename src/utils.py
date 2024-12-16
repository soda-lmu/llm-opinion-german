import os
import json
import json
import pandas as pd
import yaml 
from langdetect import detect, LangDetectException

# Base directory where you want to save the files
cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PROJECT_DIR = os.path.abspath( os.path.join(cwd))
CODING_DIR = os.path.join(PROJECT_DIR,'data','coding_values') 

def load_lookup_data(filename,DATA_DIR=CODING_DIR):
    with open(os.path.join(DATA_DIR, filename), 'r') as f:
        data= json.load(f)
        return {int(k): v for k, v in data.items()}

month_names_dict = load_lookup_data('month_names_german.json',CODING_DIR)

def format_prompt(prompt_fpath, row):
    """
    row needs age,gender,party,eastwest and year variables
    """
    start_date = pd.to_datetime(row.field_start)
    year = start_date.year
    month = month_names_dict[start_date.month]
    with open(prompt_fpath, "r", encoding="utf-8") as file:
        prompt = file.read().replace("\n", "") + "\n"
        
    artikel = "Die" if row["gender"] == "weiblich" else "Der"
    pronoun = "Sie" if row["gender"] == "weiblich" else "Er"
    pronoun2 = "Sie" if row["gender"] == "weiblich" else "Er"
    return prompt.format(
        month=month,
        artikel=artikel,
        pronoun=pronoun,
        pronoun2=pronoun2,
        age=row["age"],
        gender=row["gender"],
        party=row["leaning_party"],
        eastwest=row["ostwest"],
        year=year,
        schulabschluss_clause=row["schulabschluss_clause"],
        berufabschluss_clause=row["berufabschluss_clause"],
    )


def get_experiment_log(row, survey_wave,model_output):
    """
    Generates a dictionary containing logging information for a model's response.
    """
    log_dict= {
        "survey_wave": survey_wave,
        "user_id": row['lfdn'],
    }
    log_dict.update(model_output)
    
    return log_dict

def save_experiment_log(user_id, log_dict, experiment_dir):
    """
    Saves model answers in JSON format within an experiment subdirectory.
    """    
    filename = f"{user_id}.json"
    file_path = os.path.join(experiment_dir, filename)

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(log_dict, file, ensure_ascii=False, indent=4)
    
    print(f"Saved: {file_path}")

def translate(text,source='de', target='en'):
    from deep_translator import GoogleTranslator
    gt= GoogleTranslator(source=source, target=target)
    return gt.translate(text) 


def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_lang(text):
    try:
        language = detect(text)
        return language
        #return language
    except Exception as e:
        return ''
