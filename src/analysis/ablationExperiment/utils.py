import os
import pandas as pd

from src.analysis.data_processing import get_demographics_and_labels, get_demographics_and_llm_labels, get_wave_demographics, save_experiment_pmf,coarse_translation
from src.analysis.metrics import calculate_pmf_by_groups, calculate_pmf_population, get_cramerV_multiclass
from src.paths import RESULTS_DIR
ablation_experiments= ['1VAR_age',
 '1VAR_berufabschluss',
 '1VAR_eastwest',
 '1VAR_gender',
 '1VAR_party',
 '1VAR_schulabschluss',
 'Llama2_all',
 'Llama2_base',
 'Llama2_model_opinion',#
 'without_age',
 'without_berufabschluss',
 'without_eastwest',
 'without_gender',
 'without_party',
 'without_schulabschluss']

ablation_mapped_dict = {
        "ostwest": ["Llama2_base", "1VAR_eastwest", "without_eastwest", "Llama2_all"],
        "berufabschluss_clause": [
            "Llama2_base",
            "1VAR_berufabschluss",
            "without_berufabschluss",
            "Llama2_all",
        ],
        "leaning_party": [
            "Llama2_base",
            "1VAR_party",
            "without_party",
            "Llama2_all",
        ],
        "gender": ["Llama2_base", "1VAR_gender", "without_gender", "Llama2_all"],
        "schulabschluss_clause": [
            "Llama2_base",
            "1VAR_schulabschluss",
            "without_schulabschluss",
            "Llama2_all",
        ],
        "age_groups": [
            "Llama2_base",
            "1VAR_age",
            "without_age",
            "Llama2_all",
        ],
    }

# def get_ablationExperiment_data():
#     wave_number=12 # ablation is only performed on #12
#     demographics= get_wave_demographics(wave_number)
#     survey_labels_12 = get_demographics_and_labels(wave_number,demographics)
#     llm_labels_dict={}
#     survey_labels_dict={}
#     llm_population_pmf={}
#     llm_group_pmf={}
#     for experiment in ablation_experiments:
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




def get_ablationExperiment_data(save=False):
    wave_number=12 # ablation is only performed on #12
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


    for experiment in ablation_experiments:

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
            experiment="ablationExperiment",
        )
        save_experiment_pmf(
            survey_population_df_multiclass,
            llm_population_df_multiclass,
            survey_group_pmf_multiclass,
            llm_group_pmf_multiclass,
            method="multiclass",
            experiment="ablationExperiment",
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

def get_ablation_cramer_table(cramer_ablation_df,save=False,fname='cramer_table.csv'):

    cramer_ablation_df= cramer_ablation_df.pivot(columns=['source'],values=['Cramers\' V'],index=['index','wave_id']).reset_index()
    def filter_index(row):
        a=row['index'][0]
        exp_values= ablation_mapped_dict[a]
        wave_id=row['wave_id'][0]
        #print(a,wave_id,exp_values)
        if (wave_id in exp_values) and (wave_id!='Llama2_base') and ('without_' not in wave_id) :
            return True
        else:
            return False

    def get_experiment_type(row):
        a=row['wave_id'][0]
        if '1VAR' in a:
            return 'one variable'
        elif 'all' in a:
            return 'all variables'
        else:
            return None
        
    cramer_ablation_df['filter']=cramer_ablation_df.apply(filter_index,axis=1)     
    cramer_ablation_df=cramer_ablation_df[cramer_ablation_df['filter']==True].sort_values(by='index')        
    cramer_ablation_df['exp_type']=cramer_ablation_df.apply(get_experiment_type ,axis=1)
    cramer_ablation_df['Cramers\' V']=cramer_ablation_df['Cramers\' V'].round(3)
    cramer_ablation_df2=cramer_ablation_df.reset_index(drop=True).drop(['wave_id','filter'],axis=1).reset_index()
    cramer_ablation_df2_one=cramer_ablation_df2[cramer_ablation_df2['exp_type']=='one variable']
    cramer_ablation_df2_all=cramer_ablation_df2[cramer_ablation_df2['exp_type']=='all variables']#.query("exp_type=='one variable' ")

    c=pd.merge(cramer_ablation_df2_all,cramer_ablation_df2_one,on='index')
    c.columns = [' '.join(col).strip() for col in c.columns.values]
    c.columns=['level_0_x', 'prompt variable', 'Cramers\' V (all variables)', 'Cramers\' V (survey)',
           'exp_type_x', 'level_0_y', 'Cramers\' V (one variable)', 'Cramers\' V (survey)',
           'exp_type_y']
    c=c[[ 'prompt variable', 'Cramers\' V (survey)','Cramers\' V (all variables)','Cramers\' V (one variable)']]
    if save==True:
        c.to_csv(os.path.join(RESULTS_DIR,'ablationExperiment',fname))
    return c 


# def get_ablation_cramer_table(survey_labels_dict, llm_labels_dict,save=False):

#     cramer_ablation_df= get_cramerV_multiclass(survey_labels_dict, llm_labels_dict)
#     cramer_ablation_df= cramer_ablation_df.pivot(columns=['source'],values=['Cramers\' V'],index=['index','wave_id']).reset_index()
#     def filter_index(row):
#         a=row['index'][0]
#         exp_values= ablation_mapped_dict[a]
#         wave_id=row['wave_id'][0]
#         #print(a,wave_id,exp_values)
#         if (wave_id in exp_values) and (wave_id!='Llama2_base') and ('without_' not in wave_id) :
#             return True
#         else:
#             return False

#     def get_experiment_type(row):
#         a=row['wave_id'][0]
#         if '1VAR' in a:
#             return 'one variable'
#         elif 'all' in a:
#             return 'all variables'
#         else:
#             return None
        
#     cramer_ablation_df['filter']=cramer_ablation_df.apply(filter_index,axis=1)     
#     cramer_ablation_df=cramer_ablation_df[cramer_ablation_df['filter']==True].sort_values(by='index')        
#     cramer_ablation_df['exp_type']=cramer_ablation_df.apply(get_experiment_type ,axis=1)
#     cramer_ablation_df['Cramers\' V']=cramer_ablation_df['Cramers\' V'].round(3)
#     cramer_ablation_df2=cramer_ablation_df.reset_index(drop=True).drop(['wave_id','filter'],axis=1).reset_index()
#     cramer_ablation_df2_one=cramer_ablation_df2[cramer_ablation_df2['exp_type']=='one variable']
#     cramer_ablation_df2_all=cramer_ablation_df2[cramer_ablation_df2['exp_type']=='all variables']#.query("exp_type=='one variable' ")

#     c=pd.merge(cramer_ablation_df2_all,cramer_ablation_df2_one,on='index')
#     c.columns = [' '.join(col).strip() for col in c.columns.values]
#     c.columns=['level_0_x', 'prompt variable', 'Cramers\' V (all variables)', 'Cramers\' V (survey)',
#            'exp_type_x', 'level_0_y', 'Cramers\' V (one variable)', 'Cramers\' V (survey)',
#            'exp_type_y']
#     c=c[[ 'prompt variable', 'Cramers\' V (survey)','Cramers\' V (all variables)','Cramers\' V (one variable)']]
#     if save==True:
#         c.to_csv(os.path.join(RESULTS_DIR,'ablationExperiment','cramer_table.csv'))
#     return c 



# #compare ablation JS distances by group
# mapped_dict = {
#     "ostwest": ["Llama2_base", "1VAR_eastwest", "without_eastwest", "Llama2_all"],
#     "berufabschluss_clause": [
#         "Llama2_base",
#         "1VAR_berufabschluss",
#         "without_berufabschluss",
#         "Llama2_all",
#     ],
#     "leaning_party": [
#         "Llama2_base",
#         "1VAR_party",
#         "without_party",
#         "Llama2_all",
#     ],
#     "gender": ["Llama2_base", "1VAR_gender", "without_gender", "Llama2_all"],
#     "schulabschluss_clause": [
#         "Llama2_base",
#         "1VAR_schulabschluss",
#         "without_schulabschluss",
#         "Llama2_all",
#     ],
#     "age_groups": [
#         "Llama2_base",
#         "1VAR_age",
#         "without_age",
#         "Llama2_all",
#     ],
# }
# a_list=[]
# for category,experiments in mapped_dict.items():
#     a=group_JS[group_JS['social_group_category'].isin([category]) &group_JS['wave'].isin(experiments)]
#     sorted_experiments = {elt:i for i, elt in enumerate(experiments)}
#     #print(category,':',sorted_experiments)
#     a['wave_idx']=a['wave'].map(sorted_experiments)
#     a=a.sort_values(by=['wave'], key=lambda x: x.map(sorted_experiments) ).reset_index()
#     a_list.append(a)

# ablation_js_df_filtered= pd.concat(a_list,axis=0)
# ablation_js_df_filtered
# #get_JS_group_plot_waveExperiment2(ablation_js_df_filtered,fname='ablation_JS_by_groups.html')
