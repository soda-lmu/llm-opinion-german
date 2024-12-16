import src
import numpy as np
from src.paths import MODELS_DIR, OUTPUTS_DIR
import os
import torch
import pandas as pd
from src.data.process_data import process_open_ended, process_wave_data,process_open_ended
from src.data.read_data import load_raw_survey_data, read_stata_file
from src.paths import CODING_DIR, GLES_DIR, PROCESSED_DATA_DIR, ANNOTATED_GENERATIONS_DIR,RAW_DATA_DIR
from src.utils import get_lang
from src.bert.utils import get_experiment_df
from src.paths import RESULTS_DIR
from tqdm import tqdm



classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).upperclass_id.unique())) }
df_lookup= pd.read_csv(os.path.join(CODING_DIR,'map.csv'))
label2str= dict(zip(df_lookup.upperclass_id,df_lookup.upperclass_name))
label_names= [label2str[i] for i in range(0,len(label2str)) ]

labels_16= [label_name for label_name in label_names if label_name!='LLM refusal']
labels_14= [label_name for label_name in label_names if label_name not in ['LLM refusal' ,'keine Angabe','weiß nich'] ]

wave_dates={12: '05-11-2019',
 13: '21-04-2020',
 14: '03-11-2020',
 15: '25-02-2021',
 16: '06-05-2021',
 17: '07-07-2021',
 18: '11-08-2021',
 19: '15-09-2021',
 20: '29-09-2021',
 21: '09-12-2021'}

social_groups=[ 'ostwest','berufabschluss_clause', 'leaning_party', 'gender','schulabschluss_clause', 'age_groups']

social_category_to_group={'ostwest': ['Westdeutschland', 'Ostdeutschland'],
 'berufabschluss_clause': ['hat einen Berufsfachschulabschluss.',
  'hat einen Fachhochschulabschluss.',
  'hat einen Universitätsabschluss.',
  'hat eine kaufmännische Lehre abgeschlossen.',
  'hat einen Meisterabschluss oder Technikerabschluss.',
  'hat eine Lehre abgeschlossen.',
  'hat keine berufliche Ausbildung abgeschlossen.',
  'hat einen Fachschulabschluss.',
  'befindet sich noch in beruflicher Ausbildung.',
  'hat ein Berufliches Praktikum oder Volontariat abgeschlossen.',
  'hat eine gewerbliche oder landwirtschaftliche Lehre abgeschlossen.'],
 'leaning_party': ['die Grünen',
  'Die Linke',
  'die CDU/CSU',
  'die FDP',
  'die SPD',
  'die AfD',
  'keine Partei',
  'eine Kleinpartei'],
 'gender': ['weiblich', 'männlich'],
 'schulabschluss_clause': ['hat einen Fachhochschulreife',
  'hat das Abitur',
  'hat einen Realschulabschluss',
  'hat einen Hauptschulabschluss',
  'hat keinen Schulabschluss',
  'ist noch Schüler/in'],
 'age_groups': ['45-59 YEARS', '60 and more', '30-44 YEARS', '18-29 YEARS']}
social_group_to_category = {v: k for k, vals in social_category_to_group.items() for v in vals}

coarse_translation = {
    "Politische Strukturen und Prozesse": "Political System and Processes",
    "Sozialpolitik": "Social Policy",
    "Gesundheitspolitik": "Health Policy",
    "Familien- und Gleichstellungspolitik": "Family and Gender Equality Policy",
    "Bildungspolitik": "Education Policy",
    "Umweltpolitik": "Environmental  Policy",
    "Wirtschaftspolitik": "Economic  Policy",
    "Sicherheits": "Security",
    "Außenpolitik": "Foreign  Policy",
    "Medien und Kommunikation": "Media and  Communication",
    "Sonstiges": "Others",
    "Migration und Integration": "Migration and  Integration",
    "Ostdeutschland": "East  Germany",
    "keine Angabe": "Not specified",
    "weiß nich": "Do not know",
    "LLM refusal": "LLM refusal",
    "Werte, politische Kultur und Gesellschaftskritik": "Values, political culture and general  social criticism"
}



def get_wave_demographics(wave_number):
    wave_df, wave_open_ended_df, df_coding_840s = load_raw_survey_data(wave_number)
    wave_open_ended_df = process_open_ended(wave_open_ended_df, df_coding_840s, wave_number)
    wave_df_processed = process_wave_data(wave_df, wave_open_ended_df, wave_number).filter(regex='lfdn$|gender|^age$|^age_groups$|clause|party|ostwest|user|labels|eastwest|text|output|highest_prob_label')
    wave_df_processed['highest_prob_label']=wave_df_processed['highest_prob_label'].map(label2str)
    age_bins = [17, 29, 44, 59, 74]
    age_labels = [ '18-29 YEARS', '30-44 YEARS', '45-59 YEARS', '60 and more']
    wave_df_processed.loc[:, 'age_groups'] = pd.cut(wave_df_processed['age'], bins=age_bins, labels=age_labels, right=True)
    return wave_df_processed
    
def get_demographics_and_labels(wave_number,demographics):
    
    survey_labels_matrix=pd.DataFrame(demographics.labels.tolist(), columns=label_names  )
    survey_labels_matrix.index=demographics.lfdn
    survey_labels_matrix.drop(list(survey_labels_matrix.filter(regex='LLM refusal')), axis=1, inplace=True)
    k=pd.merge(demographics, survey_labels_matrix, left_on='lfdn',right_on=survey_labels_matrix.index, how='inner')#.groupby('ostwest').apply(lambda x:)
    return k

def extract_model_predictions(row, model_name):
    try:
        for entry in row:
            if entry['model_name'] == model_name:
                pred = entry['result'].get('predictions', None)
                return pred
    except:
        print(row, model_name)

def get_highest_prob_label(row, model_name):
    try:
        for entry in row:
            if entry['model_name'] == model_name:
                classification_result = entry['result']
                label_names = classification_result['pred_label_names']
                label_probs = classification_result['pred_label_probs']

                max_prob_index = label_probs.index(max(label_probs))

                # Return the label name with the highest probability
                return label_names[max_prob_index]
        return None 
    except:
        print(row, model_name)
        


def get_langs(text):
    from lingua import Language, LanguageDetectorBuilder
    languages = [Language.ENGLISH,  Language.GERMAN]

    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    det_langs=[]
    for result in detector.detect_multiple_languages_of(text):
        #print(f"{result.language.name}: '{sentence[result.start_index:result.end_index]}'")
        det_langs.append(result.language.name)
    return '_'.join(sorted(list(set(det_langs))))
        
def get_demographics_and_llm_labels(wave_number,experiment_folder,wave_demographics):
    experiment_path=os.path.join(OUTPUTS_DIR,'text_generations',str(wave_number),experiment_folder)
    print(experiment_path)
    df_exp= get_experiment_df(experiment_path)
    
    df_exp= df_exp[df_exp['user_id'].isin(wave_demographics['lfdn'].tolist() )]# to make sure numbers match,i.e  we dont put GLES-LLM inconclusive answers 
    
    model_preds = df_exp['classification_coarse'].apply(lambda x: extract_model_predictions(x, model_name='bert_mixed_coarse_resample20240708_195103'))
    llm_labels_matrix=  pd.DataFrame(model_preds.tolist(), columns=label_names )
    llm_labels_matrix.index = df_exp['user_id']
    llm_labels_matrix.drop(list(llm_labels_matrix.filter(regex='LLM refusal')), axis=1, inplace=True)
    
    highest_prob_pred= df_exp['classification_coarse'].apply(lambda x: get_highest_prob_label(x, model_name='bert_mixed_coarse_resample20240708_195103'))
    llm_labels_matrix['highest_prob_label_llm']=highest_prob_pred.values
    llm_labels_matrix['text_llm']= df_exp['output'].values
    llm_labels_matrix['text_llm_lang']=llm_labels_matrix['text_llm'].apply(get_langs).values
    k2=pd.merge(wave_demographics, llm_labels_matrix, left_on='lfdn',right_on=llm_labels_matrix.index, how='inner')#.groupby('ostwest').apply(lambda x:)
    return k2

def concat_colnames_nonzero(row):
    return '_'.join([col for col in row.index if row[col] != 0])




def sample_random_label_from_strata(df):
    df= df[['ostwest',
     'berufabschluss_clause',
     'leaning_party',
     'gender',
     'schulabschluss_clause',
     'age_groups',
     'highest_prob_label']]

    import pandas as pd
    import numpy as np

    # Assuming your dataframe is named 'df'

    # Step 1: Group by all columns except highest_prob_label and highest_prob_label_sampled
    grouping_columns = [col for col in df.columns if col not in ['highest_prob_label', 'highest_prob_label_sampled']]

    # Step 2 & 3: Get unique values and their frequencies, create probability distribution
    def sample_label(group):
        value_counts = group['highest_prob_label'].value_counts(normalize=True)
        return pd.Series(np.random.choice(value_counts.index, p=value_counts.values, size=len(group)), index=group.index)

    # Step 4: Sample a new value for each group
    df['new_sampled_label'] = df.groupby(grouping_columns, group_keys=False).apply(sample_label)
    return df 
# Display the result



def save_experiment_pmf(
    survey_population_df,
    llm_population_df,
    survey_group_pmf,
    llm_group_pmf,
    method="multilabel",
    experiment=None,
):
    if experiment is None:
        raise ValueError("experiment should be waveExperiment, ablationExperiment or modelExperiment")
    
    survey_population_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            experiment,
            f"survey_population_level_pmf_{method}.csv",
        ),
        index=True,
    )
    llm_population_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            experiment,
            f"llm_population_level_pmf_{method}.csv",
        ),
        index=True,
    )

    for (
        key,
        df,
    ) in (
        survey_group_pmf.items()
    ):  # saving social group level pmf of each wave seperately.
        fname = f"survey_group_pmf_{key}_{method}.csv"
        df.to_csv(os.path.join(RESULTS_DIR, experiment, fname), index=True)

    for key, df in llm_group_pmf.items():
        fname = f"llm_group_pmf_{key}_{method}.csv"
        df.to_csv(os.path.join(RESULTS_DIR, experiment, fname), index=True)

