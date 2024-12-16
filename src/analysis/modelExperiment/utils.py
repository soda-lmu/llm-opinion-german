import pandas as pd

from src.analysis.data_processing import get_demographics_and_labels, get_demographics_and_llm_labels, get_wave_demographics, save_experiment_pmf,coarse_translation
from src.analysis.metrics import calculate_pmf_by_groups, calculate_pmf_population

model_experiments= [
 'google-gemma-7b-it_12_1712704376_modified',
 'Llama2_all',
 'mistralai-Mixtral-8x7B-Instruct-v0.1_12_1712772173',
 #'Llama3_70B_all'
]


def get_modelExperiment_data(save=False):
    wave_number=12 # model exp is only performed on #12
    demographics= get_wave_demographics(wave_number)
    survey_labels_12 = get_demographics_and_labels(wave_number,demographics)

    survey_labels_dict = {}
    llm_labels_dict = {}

    survey_population_pmf_multilabel = {}
    llm_population_pmf_multilabel = {}
    survey_group_pmf_multilabel = {}
    llm_group_pmf_multilabel = {}

    survey_population_pmf_multiclass = {}
    llm_population_pmf_multiclass = {}
    survey_group_pmf_multiclass = {}
    llm_group_pmf_multiclass = {}


    for experiment in model_experiments:
        # demographics = get_wave_demographics(wave_number)

        # survey_labels = get_demographics_and_labels(wave_number, demographics)
        llm_labels = get_demographics_and_llm_labels(
            wave_number, experiment, demographics
        )

        survey_labels_dict[experiment] = survey_labels_12
        llm_labels_dict[experiment] = llm_labels

        survey_population_pmf_multilabel[experiment] = calculate_pmf_population(
            survey_labels_12, method="multilabel"
        )
        llm_population_pmf_multilabel[experiment] = calculate_pmf_population(
            llm_labels, method="multilabel"
        )
        survey_group_pmf_multilabel[experiment] = calculate_pmf_by_groups(
            survey_labels_12, method="multilabel"
        ).rename(coarse_translation,axis=1)
        llm_group_pmf_multilabel[experiment] = calculate_pmf_by_groups(
            llm_labels, method="multilabel"
        ).rename(coarse_translation,axis=1)

        survey_population_pmf_multiclass[experiment] = calculate_pmf_population(
            survey_labels_12, method="multiclass"
        )
        llm_population_pmf_multiclass[experiment] = calculate_pmf_population(
            llm_labels, method="multiclass"
        )
        survey_group_pmf_multiclass[experiment] = calculate_pmf_by_groups(
            survey_labels_12, method="multiclass"
        ).rename(coarse_translation,axis=1)
        llm_group_pmf_multiclass[experiment] = calculate_pmf_by_groups(
            llm_labels, method="multiclass"
        ).rename(coarse_translation,axis=1)

    survey_population_df_multilabel = pd.DataFrame(survey_population_pmf_multilabel).rename(coarse_translation,axis=0)
    llm_population_df_multilabel = pd.DataFrame(llm_population_pmf_multilabel).rename(coarse_translation,axis=0)

    survey_population_df_multiclass = pd.DataFrame(survey_population_pmf_multiclass).rename(coarse_translation,axis=0)
    llm_population_df_multiclass = pd.DataFrame(llm_population_pmf_multiclass).rename(coarse_translation,axis=0)

    if save == True:
        save_experiment_pmf(
            survey_population_df_multilabel,
            llm_population_df_multilabel,
            survey_group_pmf_multilabel,
            llm_group_pmf_multilabel,
            method="multilabel",
            experiment="modelExperiment",
        )
        save_experiment_pmf(
            survey_population_df_multiclass,
            llm_population_df_multiclass,
            survey_group_pmf_multiclass,
            llm_group_pmf_multiclass,
            method="multiclass",
            experiment="modelExperiment",
        )

    return (
        survey_labels_dict,
        llm_labels_dict,
        # multilabel
        survey_population_df_multilabel,  # df
        llm_population_df_multilabel,  # df
        survey_group_pmf_multilabel,  # dict of dfs
        llm_group_pmf_multilabel,  # dict of dfs
        # multiclass
        survey_population_df_multiclass,  # df
        llm_population_df_multiclass,  # df
        survey_group_pmf_multiclass,  # dict of dfs
        llm_group_pmf_multiclass,  # dict of dfs
    )

from src.analysis.data_processing import labels_16,concat_colnames_nonzero

def get_textual_stats(llm_labels_dict1,survey_labels_dict1):
    d={}
    llm_labels_dict1['survey'] = survey_labels_dict1[list(survey_labels_dict1.keys())[0]] #any key from survey_labels_dict1 will work, they are repeetitions just to match llm_labels_dict

    for k,df in llm_labels_dict1.items():
        avg_label_cnt=df[labels_16].sum(axis=1).mean()
        avg_sample_per_label= df[labels_16].sum(axis=0).mean()
        if k =='survey':
            avg_word_count=df['text'].apply(lambda x: len(x.split())).mean()
        else:
            avg_word_count=df['text_llm'].apply(lambda x: len(x.split())).mean()
        labels_concatted= df[labels_16].apply(concat_colnames_nonzero,axis=1)
        labels_concatted= labels_concatted[labels_concatted.str.contains("_")]
        lbl_vc= labels_concatted.value_counts(1).head(5)
        d[k]={
            'avg_label_cnt':avg_label_cnt,
            'avg_sample_per_label':avg_sample_per_label,
            'avg_word_count':avg_word_count,
        }
    df= pd.DataFrame(d).round(2)#.to_csv()
    return df 
    
# def get_modelExperiment_data():
#     wave_number=12 # ablation is only performed on #12
#     demographics= get_wave_demographics(wave_number)
#     survey_labels_12 = get_demographics_and_labels(wave_number,demographics)
#     llm_labels_dict={}
#     survey_labels_dict={}
#     llm_population_pmf={}
#     llm_group_pmf={}
#     for experiment in model_experiments:
#         llm_labels = get_demographics_and_llm_labels(wave_number,experiment,demographics)
#         llm_labels_dict[experiment]=llm_labels
#         llm_population_pmf[experiment] =     calculate_pmf_population(llm_labels)
#         llm_group_pmf[experiment] =     calculate_pmf_by_groups(llm_labels)
                
        
#     survey_population_pmf =  calculate_pmf_population(survey_labels_12)
#     survey_group_pmf_wave_12 =  calculate_pmf_by_groups(survey_labels_12)
#     survey_population_df=pd.DataFrame(survey_population_pmf)
#     llm_population_df = pd.DataFrame(llm_population_pmf)

#     survey_labels_dict={}
#     for key in llm_labels_dict.keys():
#         survey_labels_dict[key]=survey_labels_12
        
#     repeated_df = pd.concat([survey_population_df] * llm_population_df.shape[1], axis=1)
#     repeated_df.columns=llm_population_df.columns
#     survey_population_df= repeated_df

#     survey_group_pmf={}
#     for key in llm_group_pmf.keys():

#         survey_group_pmf[key]=survey_group_pmf_wave_12

#     return survey_labels_dict,llm_labels_dict,survey_population_df,survey_group_pmf,llm_population_df,llm_group_pmf

# # ablation_experiments=['1VAR_age']
# def get_modelExperiment_data():
#     wave_number=12 # modelExperiment is only performed on #12
#     demographics= get_wave_demographics(wave_number)
    
#     survey_labels_12 = get_demographics_and_labels(wave_number,demographics)
#     llm_labels_dict={}
#     llm_population_pmf={}
#     llm_group_pmf={}

#     for experiment in model_experiments:
#         llm_labels = get_demographics_and_llm_labels(wave_number,experiment,demographics)
#         llm_labels_dict[experiment]=llm_labels
#         llm_population_pmf[experiment] =     calculate_pmf_population(llm_labels)
#         llm_group_pmf[experiment] =     calculate_pmf_by_groups(llm_labels)

        
#     survey_population_pmf =  calculate_pmf_population(survey_labels_12)
#     survey_group_pmf =  calculate_pmf_by_groups(survey_labels_12)
#     survey_population_df=pd.DataFrame(survey_population_pmf)
#     llm_population_df = pd.DataFrame(llm_population_pmf)

#     survey_labels_dict={}
#     for key in llm_labels_dict.keys():
#         survey_labels_dict[key]=survey_labels_12
        
#     repeated_df = pd.concat([survey_population_df] * llm_population_df.shape[1], axis=1)
#     repeated_df.columns=llm_population_df.columns
#     survey_population_df= repeated_df
#     return survey_labels_dict,llm_labels_dict,survey_population_df,survey_group_pmf,llm_population_df,llm_group_pmf

