import os

import pandas as pd
import time
from src.analysis.data_processing import (
    concat_colnames_nonzero,
    get_demographics_and_labels,
    get_demographics_and_llm_labels,
    get_wave_demographics,
    labels_16,
    save_experiment_pmf,
)
from src.analysis.metrics import (
    calculate_cramerV,
    calculate_group_entropy,
    calculate_pmf_by_groups,
    calculate_pmf_population,
    calculate_population_entropy,
    get_MI_from_dataset,
    calculate_cramerV_multiclass,
    get_js_dist_by_groups,
    get_js_dist_population,
)
from src.paths import RESULTS_DIR


def get_experiment_entropy(
    
    survey_population_df, llm_population_df, survey_group_pmf, llm_group_pmf
):

    def combine_results(survey_dict, llm_dict):
        df1 = pd.DataFrame([survey_dict]).T
        df1["study"] = df1.index
        df1["source"] = "survey"

        df2 = pd.DataFrame([llm_dict]).T
        df2["study"] = df2.index
        df2["source"] = "llm"
        # Concatenate the DataFrames
        combined_results = pd.concat([df1, df2], ignore_index=True)
        combined_results = combined_results.rename({0: "shannon_entropy"}, axis=1)
        return combined_results

    def get_dict_to_df(json_data, source):
        data_list = []
        for wave_id, social_groups in json_data.items():
            for social_group, entropy in social_groups.items():
                data_list.append([wave_id, social_group, entropy])
        df = pd.DataFrame(data_list, columns=["wave_id", "social_group", "entropy"])
        df["source"] = source
        return df

    ###population-level results
    survey_population_entropy = calculate_population_entropy(survey_population_df)
    llm_population_entropy = calculate_population_entropy(llm_population_df)
    population_level_entropy_results = combine_results(
        survey_population_entropy, llm_population_entropy
    )
    ###population-level results

    ###group-level results
    survey_group_entropy_dict = {}
    llm_group_entropy_dict = {}
    assert survey_group_pmf.keys() == llm_group_pmf.keys()
    for i in survey_group_pmf.keys():
        survey_group_entropy_dict[i] = calculate_group_entropy(survey_group_pmf[i])
        llm_group_entropy_dict[i] = calculate_group_entropy(llm_group_pmf[i])
    group_level_entropy_results = pd.concat(
        [
            get_dict_to_df(survey_group_entropy_dict, source="survey"),
            get_dict_to_df(llm_group_entropy_dict, source="llm"),
        ],
        ignore_index=True,
    )
    ###group-level results

    return population_level_entropy_results, group_level_entropy_results

def get_JS_experiment(
    survey_population_df, llm_population_df, survey_group_pmf, llm_group_pmf
):

    def get_groupedJS_waveExperiment(survey_group_pmf, llm_group_pmf):
        print(survey_group_pmf.keys(), llm_group_pmf.keys())
        assert survey_group_pmf.keys() == llm_group_pmf.keys()
        js_dfs = []
        for i in survey_group_pmf.keys():
            df = get_js_dist_by_groups(survey_group_pmf[i], llm_group_pmf[i])
            df.loc[:, "wave"] = i
            js_dfs.append(df)

        return pd.concat(js_dfs)

    group_JS = get_groupedJS_waveExperiment(survey_group_pmf, llm_group_pmf)
    population_JS = get_js_dist_population(survey_population_df, llm_population_df)
    return population_JS, group_JS



def get_MI_experiment(survey_labels_dict, llm_labels_dict):
    wave_ids = survey_labels_dict.keys()
    dfs = []
    for wave_id in wave_ids:
        mi_survey = get_MI_from_dataset(survey_labels_dict[wave_id])
        mi_survey["wave"] = wave_id
        mi_survey["source"] = "survey"
        mi_llm = get_MI_from_dataset(llm_labels_dict[wave_id])
        mi_llm["wave"] = wave_id
        mi_llm["source"] = "llm"
        dfs.append(mi_survey)
        dfs.append(mi_llm)

    return pd.concat(dfs)



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
    entropy_JS_corr_data=k
    return entropy_JS_corr_data
