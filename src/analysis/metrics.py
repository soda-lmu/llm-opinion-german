import os
from src.analysis.data_processing import get_wave_demographics, get_demographics_and_labels, extract_model_predictions, get_highest_prob_label
from src.analysis.data_processing import social_groups,label_names,labels_16,labels_14,label2str,df_lookup,social_group_to_category
from scipy.spatial import distance

from src.analysis.data_processing import concat_colnames_nonzero
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial import distance
from scipy.stats import entropy
from sklearn.metrics import cohen_kappa_score, accuracy_score
from dython.nominal import associations
import numpy as np

from src.paths import RESULTS_DIR

def calculate_pmf_population(df,method='multilabel'):
    if method=='multilabel':
        filtered_columns = [col for col in df.columns if col in labels_14]
        column_sum= df[filtered_columns].sum(axis=0)
        dist=column_sum/column_sum.sum().round(4)
    elif method=='multiclass':
        
        if 'highest_prob_label_llm' in df.columns:# calculate for llm dataset
            col='highest_prob_label_llm'
        elif 'highest_prob_label' in df.columns:# calculate for survey dataset
            col='highest_prob_label'
        else:
            raise ValueError('No label column found, dataframe must contain either highest_prob_label_llm or highest_prob_label')
        
        df = df[df[col].isin(labels_14)] # remove no answe or dont know answers
        dist = df[col].value_counts(1).round(4)
        dist= dist.reindex(labels_14, fill_value=0)

    else:
        raise ValueError('method must be either multilabel or multiclass')
    return dist

def calculate_pmf_by_groups(df,method='multilabel'):
    if method not in ['multilabel','multiclass']:
        raise ValueError('method must be either multilabel or multiclass')

    if method=='multilabel':
        filtered_columns = [col for col in df.columns if col in labels_14]
        dist_list=[]
        for key in social_groups:
            grouped = df.groupby(key)
            sum_lbl = grouped[filtered_columns].sum()
            dist = sum_lbl.div(sum_lbl.sum(axis=1),axis=0).round(4)
            dist_list.append(dist)
        
        dist_df= pd.concat(dist_list).fillna(0)
    elif method=='multiclass':
        if 'highest_prob_label_llm' in df.columns:
            col='highest_prob_label_llm'
        elif 'highest_prob_label' in df.columns:
            col='highest_prob_label'
        else:
            raise ValueError('No label column found, dataframe must contain either highest_prob_label_llm or highest_prob_label')
        dist_list=[]
        df = df[df[col].isin(labels_14)] # remove no answe or dont know answers
        for key in social_groups:
            grouped = df.groupby(key)
            dist = grouped[col].value_counts(1).unstack().round(4)
            for c in labels_14:
                if c not in dist.columns:
                    #print(f'{key},{c} with 0')
                    dist[c] = 0
            dist = dist[labels_14]
            #dist= dist.reindex(label_names, fill_value=0) # fill missing labels with 0
            dist_list.append(dist)
        dist_df= pd.concat(dist_list).fillna(0)
    return dist_df


def get_js_dist_population(df1,df2):
    if type(df1)==pd.DataFrame:
        assert (df1.columns==df2.columns).all()
    
        js_arr=distance.jensenshannon(df1,df2).tolist()
        df= pd.DataFrame([js_arr],columns=df1.columns)
    return  df

def get_js_dist_by_groups(df1,df2):
    assert df1.shape[0]==df2.shape[0]
    js_dist_list=[]
    for i in range(0,df1.shape[0]):
            df1_group_i = df1.iloc[i]
            df2_group_i = df2.iloc[i]
            df1_groupname=df1_group_i.name
            df2_groupname=df2_group_i.name
            assert df1_groupname ==df2_groupname
            js=distance.jensenshannon(df1_group_i,df2_group_i)
            js_dist_list.append([df1_groupname,js])
    df=pd.DataFrame(js_dist_list,columns=['social_group','js'])
    df['social_group_category']=df['social_group'].map(social_group_to_category)
    return  df



            
def get_MI_from_dataset( df ):
    if 'highest_prob_label_llm' in df.columns:
        col='highest_prob_label_llm'
    elif 'highest_prob_label' in df.columns:
        col='highest_prob_label'

    #df.loc[:, 'labels_concatted'] = df[labels_14].apply(concat_colnames_nonzero,axis=1)
    df= df.filter(regex=f'gender|^age_groups$|clause|party|ostwest|{col}|eastwest')
    df['combined_features'] = df.apply(lambda row: '_'.join(row.drop(col).astype(str)), axis=1)
    data = df

    label_encoder = LabelEncoder()
    df_encoded = df.apply(label_encoder.fit_transform)

    X = df_encoded#.drop('first_label_categorical', axis=1)
    y = df_encoded[col]

    mi_scores = mutual_info_classif(X, y,discrete_features=True)
    mi_results = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})

    #mi_results = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores/mi_scores.max()})
    return mi_results

def calculate_cramerV(df):
    df.loc[:, "labels_concatted"] = df[labels_14].apply(concat_colnames_nonzero, axis=1)
    df = df.filter(
        regex="gender|^age_groups$|clause|party|ostwest|labels_concatted|eastwest"
    )
    # df['combined_features'] = df.apply(lambda row: '_'.join(row.drop('labels_concatted').astype(str)), axis=1)
    return associations(df, compute_only=True)["corr"]["labels_concatted"]


def calculate_cramerV_multiclass(df, y_col):
    df = df.filter(regex=f"gender|^age_groups$|clause|party|ostwest|{y_col}|eastwest")
    return associations(df, compute_only=True)["corr"][f"{y_col}"]

# def calculate_JS_as_multiclass(df):
#     a = df["highest_prob_label_llm"].value_counts(1)
#     b = df["highest_prob_label"].value_counts(1)
#     c = pd.concat([a, b], axis=1).fillna(0)
#     return distance.jensenshannon(c.iloc[:, 0], c.iloc[:, 1])

def get_population_acc_ck(llm_labels_dict):
    for i in llm_labels_dict.keys():
        acc = accuracy_score(
            llm_labels_dict[i]["highest_prob_label"],
            llm_labels_dict[i]["highest_prob_label_llm"],
        )
        ck = cohen_kappa_score(
            llm_labels_dict[i]["highest_prob_label"],
            llm_labels_dict[i]["highest_prob_label_llm"],
        )
        print(acc, ck)


def calculate_population_entropy(df):
        entropy_dict = {}
        for col in df.columns:
            H = entropy(df[col], base=2)
            entropy_dict[col] = H
        return entropy_dict

def calculate_group_entropy(df):
        return dict(zip(df.index, entropy(df, base=2, axis=1)))



def get_cramerV_multiclass(survey_labels_dict, llm_labels_dict):
    results = []
    for wave_id in survey_labels_dict.keys():
        s_cramer = pd.DataFrame(
            calculate_cramerV_multiclass(survey_labels_dict[wave_id], y_col="highest_prob_label")
        ).rename({"highest_prob_label": "Cramers' V"}, axis=1)
        s_cramer["source"] = "survey"
        s_cramer["wave_id"] = wave_id

        l_cramer = pd.DataFrame(
            calculate_cramerV_multiclass(llm_labels_dict[wave_id], y_col="highest_prob_label_llm")
        ).rename({"highest_prob_label_llm": "Cramers' V"}, axis=1)
        l_cramer["source"] = "llm"
        l_cramer["wave_id"] = wave_id

        results.append(s_cramer)
        results.append(l_cramer)

    results_df = pd.concat(results)
    results_df = results_df.reset_index()
    results_df = results_df.query("index!='highest_prob_label_llm'").query(
        "index!='highest_prob_label'"
    )
    return results_df


def get_cramerV(survey_labels_dict, llm_labels_dict):
    results = []
    for wave_id in survey_labels_dict.keys():
        s_cramer = pd.DataFrame(calculate_cramerV(survey_labels_dict[wave_id])).rename(
            {"labels_concatted": "Cramers' V"}, axis=1
        )
        s_cramer["source"] = "survey"
        s_cramer["wave_id"] = wave_id

        l_cramer = pd.DataFrame(calculate_cramerV(llm_labels_dict[wave_id])).rename(
            {"labels_concatted": "Cramers' V"}, axis=1
        )
        l_cramer["source"] = "llm"
        l_cramer["wave_id"] = wave_id

        results.append(s_cramer)
        results.append(l_cramer)

    results_df = pd.concat(results)
    results_df = results_df.reset_index()
    results_df = results_df.query("index!='labels_concatted'")
    return results_df


def get_entropy_JS_corr_data(group_JS,group_level_entropy_results):
    shortened_dict = {#for plotting
        '18-29 YEARS': '18-29',
        '30-44 YEARS': '30-44',
        '45-59 YEARS': '45-59',
        '60 and more': '60+',
        'Die Linke': 'Linke',
        'Ostdeutschland': 'Ostdeutschland',
        'Westdeutschland': 'Westdeutschland',
        'befindet sich noch in beruflicher Ausbildung.': 'beruflicher Ausbildung',
        'die AfD': 'AfD',
        'die CDU/CSU': 'CDU/CSU',
        'die FDP': 'FDP',
        'die Grünen': 'Grünen',
        'die SPD': 'SPD',
        'eine Kleinpartei': 'Kleinpartei',
        'hat das Abitur': 'Abitur',
        'hat ein Berufliches Praktikum oder Volontariat abgeschlossen.': 'Berufliches Praktikum/Volontariat',
        'hat eine Lehre abgeschlossen.': 'Lehre abgeschlossen',
        'hat eine gewerbliche oder landwirtschaftliche Lehre abgeschlossen.': 'gewerbliche/landwirtschaftliche Lehre',
        'hat eine kaufmännische Lehre abgeschlossen.': 'kaufmännische Lehre',
        'hat einen Berufsfachschulabschluss.': 'Berufsfachschulabschluss',
        'hat einen Fachhochschulabschluss.': 'Fachhochschulabschluss',
        'hat einen Fachhochschulreife': 'Fachhochschulreife',
        'hat einen Fachschulabschluss.': 'Fachschulabschluss',
        'hat einen Hauptschulabschluss': 'Hauptschulabschluss',
        'hat einen Meisterabschluss oder Technikerabschluss.': 'Meister-/Technikerabschluss',
        'hat einen Realschulabschluss': 'Realschulabschluss',
        'hat einen Universitätsabschluss.': 'Universitätsabschluss',
        'hat keine berufliche Ausbildung abgeschlossen.': 'keine berufliche Ausbildung',
        'hat keinen Schulabschluss': 'kein Schulabschluss',
        'keine Partei': 'keine Partei',
        'männlich': 'männlich',
        'weiblich': 'weiblich'
    }
    k=pd.merge(group_JS,group_level_entropy_results.query("source=='survey'"),left_on=['wave','social_group'],right_on=['wave_id','social_group'])
    k=k.groupby(['social_group','social_group_category'])[['js','entropy']].mean().reset_index()
    k=k.query("social_group!='ist noch Schüler/in' ")
    k.loc[:,'text']=k['social_group'].map(shortened_dict)
    k=k.sort_values(by='entropy')

    
    k['social_group'].map(shortened_dict)
    return k
    
def get_entropy_JS_corr_data_no_mean(group_JS,group_level_entropy_results):
    shortened_dict = {#for plotting
        '18-29 YEARS': '18-29',
        '30-44 YEARS': '30-44',
        '45-59 YEARS': '45-59',
        '60 and more': '60+',
        'Die Linke': 'Linke',
        'Ostdeutschland': 'Ostdeutschland',
        'Westdeutschland': 'Westdeutschland',
        'befindet sich noch in beruflicher Ausbildung.': 'beruflicher Ausbildung',
        'die AfD': 'AfD',
        'die CDU/CSU': 'CDU/CSU',
        'die FDP': 'FDP',
        'die Grünen': 'Grünen',
        'die SPD': 'SPD',
        'eine Kleinpartei': 'Kleinpartei',
        'hat das Abitur': 'Abitur',
        'hat ein Berufliches Praktikum oder Volontariat abgeschlossen.': 'Berufliches Praktikum/Volontariat',
        'hat eine Lehre abgeschlossen.': 'Lehre abgeschlossen',
        'hat eine gewerbliche oder landwirtschaftliche Lehre abgeschlossen.': 'gewerbliche/landwirtschaftliche Lehre',
        'hat eine kaufmännische Lehre abgeschlossen.': 'kaufmännische Lehre',
        'hat einen Berufsfachschulabschluss.': 'Berufsfachschulabschluss',
        'hat einen Fachhochschulabschluss.': 'Fachhochschulabschluss',
        'hat einen Fachhochschulreife': 'Fachhochschulreife',
        'hat einen Fachschulabschluss.': 'Fachschulabschluss',
        'hat einen Hauptschulabschluss': 'Hauptschulabschluss',
        'hat einen Meisterabschluss oder Technikerabschluss.': 'Meister-/Technikerabschluss',
        'hat einen Realschulabschluss': 'Realschulabschluss',
        'hat einen Universitätsabschluss.': 'Universitätsabschluss',
        'hat keine berufliche Ausbildung abgeschlossen.': 'keine berufliche Ausbildung',
        'hat keinen Schulabschluss': 'kein Schulabschluss',
        'keine Partei': 'keine Partei',
        'männlich': 'männlich',
        'weiblich': 'weiblich'
    }
    k=pd.merge(group_JS,group_level_entropy_results.query("source=='survey'"),left_on=['wave','social_group'],right_on=['wave_id','social_group'])
    k=k[['social_group','social_group_category','js','entropy']]
    k=k.query("social_group!='ist noch Schüler/in' ")
    k.loc[:,'text']=k['social_group'].map(shortened_dict)
    k=k.sort_values(by='entropy')

    
    k['social_group'].map(shortened_dict)
    return k
def proportional_agreement(df1, df2):
    if set(df1.columns) != set(df2.columns):
        raise ValueError("Dataframes must have the same columns")
    if len(df1) != len(df2):
        raise ValueError("Dataframes must have the same number of rows")
    agreement = {}
    for col in df1.columns:
        agree = (df1[col] == df2[col]).sum()
        prop_agr = agree / len(df1)
        agreement[col] = prop_agr
    
    return agreement

def get_population_level_ape_results(survey_population_df_multiclass1,llm_population_df_multiclass1,survey_population_df_multilabel1,llm_population_df_multilabel1,save=False,experiment_type=None,file_name='ape_results.csv'):
    
    def get_ape(survey_population_df,llm_population_df):
        return np.abs(survey_population_df - llm_population_df).sum()*100
        
    ape_multiclass = get_ape(survey_population_df_multiclass1,llm_population_df_multiclass1)
    ape_multilabel = get_ape(survey_population_df_multilabel1,llm_population_df_multilabel1)
    ape_results= pd.concat([ape_multiclass, ape_multilabel], axis=1, keys=['multiclass', 'multilabel'])
    if save and experiment_type:
        ape_results.to_csv(os.path.join(RESULTS_DIR, experiment_type, file_name))
    return ape_results

