import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PROMPT_DIR = os.path.join(DATA_DIR, 'prompts')

GLES_DIR=  os.path.join(RAW_DATA_DIR,"GLES") 
CODING_DIR = os.path.join(PROJECT_DIR,'data','coding_values') 
MODELS_DIR = os.path.join(PROJECT_DIR,'models')
OPEN_ENDED_GLES_DR = os.path.join(GLES_DIR, 'open_ended')

OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'outputs')
TRAINING_OUT_DIR = os.path.join(OUTPUTS_DIR, 'clf_training')

GENERATIONS_DIR = os.path.join(OUTPUTS_DIR, 'text_generations')
ANNOTATED_GENERATIONS_DIR = os.path.join(PROCESSED_DATA_DIR, 'annotated_generations')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')
CLSF_LOGS_DIR = os.path.join(LOGS_DIR, 'classification_reports')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'bert_train_data')
RESULTS_DIR =   os.path.join(PROJECT_DIR, 'results')
TEXT_GEN_DIR = os.path.join(OUTPUTS_DIR, 'text_generations')

if __name__ == "__main__":
    print("PROJECT_DIR:",PROJECT_DIR)
    print("RAW_DATA_DIR:",RAW_DATA_DIR)
    print("PROCESSED_DATA_DIR:",PROCESSED_DATA_DIR)
    print("GLES_DIR:",GLES_DIR)
    print("CODING_DIR:",CODING_DIR)
    print("PROMPT_DIR:",PROMPT_DIR)
    print("MODELS_DIR:",MODELS_DIR)
    print("OUTPUTS_DIR:",OUTPUTS_DIR)
    print("TRAINING_OUT_DIR:",TRAINING_OUT_DIR)
    print("GENERATIONS_DIR:",GENERATIONS_DIR)


