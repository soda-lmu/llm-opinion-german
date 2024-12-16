import time
import pandas as pd

from src.analysis.data_processing import get_demographics_and_labels, get_demographics_and_llm_labels, get_wave_demographics
from src.analysis.experiment_utils import get_JS_experiment, get_MI_experiment, get_experiment_entropy
from src.analysis.metrics import calculate_pmf_by_groups, calculate_pmf_population, get_entropy_JS_corr_data,get_cramerV, get_cramerV_multiclass, get_population_level_ape_results
from src.analysis.waveExperiment.utils import   get_waveExperiment_data

from src.analysis.waveExperiment.plots import *

begin = time.time()
(
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
) = get_waveExperiment_data(until=22)
end = time.time()
print(end - begin)



method = "multilabel"
survey_population_df, llm_population_df, survey_group_pmf, llm_group_pmf = (
    survey_population_df_multilabel,
    llm_population_df_multilabel,
    survey_group_pmf_multilabel,
    llm_group_pmf_multilabel,
)

MI_results_waveExperiment = get_MI_experiment(survey_labels_dict, llm_labels_dict)
population_JS, group_JS = get_JS_experiment(
    survey_population_df, llm_population_df, survey_group_pmf, llm_group_pmf
)
population_level_entropy_results, group_level_entropy_results = (
    get_experiment_entropy(
        survey_population_df, llm_population_df, survey_group_pmf, llm_group_pmf
    )
)
# cramer_results = get_cramerV(survey_labels_dict, llm_labels_dict)
# cramer_results_multiclass = get_cramerV_multiclass(
#     survey_labels_dict, llm_labels_dict
# )
#ape_results= get_population_level_ape_results(survey_population_df,llm_population_df,survey_population_df,llm_population_df,save=True,experiment_type='waveExperiment',file_name='ape_results_multilabel.csv')
get_JS_group_plot_waveExperiment(group_JS, fname="JS_group_plot_multilabel.html")
get_survey_to_survey_JS_distances(survey_population_df,fname='s2s_JS_dist_multilabel.html')

method="multiclass"

survey_population_df, llm_population_df, survey_group_pmf, llm_group_pmf = (
    survey_population_df_multiclass,  # df
    llm_population_df_multiclass,  # df
    survey_group_pmf_multiclass,  # dict of dfs
    llm_group_pmf_multiclass,  # dict of dfs
)

MI_results_waveExperiment = get_MI_experiment(survey_labels_dict, llm_labels_dict)
population_JS, group_JS = get_JS_experiment(
    survey_population_df, llm_population_df, survey_group_pmf, llm_group_pmf
)
population_level_entropy_results, group_level_entropy_results = (
    get_experiment_entropy(
        survey_population_df, llm_population_df, survey_group_pmf, llm_group_pmf
    )
)
cramer_results = get_cramerV(survey_labels_dict, llm_labels_dict)
cramer_results_multiclass = get_cramerV_multiclass(
    survey_labels_dict, llm_labels_dict
)
entropy_JS_corr_data=get_entropy_JS_corr_data(group_JS,group_level_entropy_results)

visualize_cramer_waveExperiment_1x6(cramer_results,fname='cramer_waveExperiment_multilabel_1x6.html')
visualize_cramer_waveExperiment_1x6(cramer_results_multiclass,fname='cramer_waveExperiment_multiclass_1x6.html')

#ape_results= get_population_level_ape_results(survey_population_df,llm_population_df,survey_population_df,llm_population_df,save=True,experiment_type='waveExperiment',file_name='ape_results_multiclass.csv')
get_JS_group_plot_waveExperiment(group_JS, fname="JS_group_plot_multiclass.html")
get_survey_to_survey_JS_distances(survey_population_df,fname='s2s_JS_dist_multiclass.html')

entropy_JS_corr_data= get_entropy_JS_corr_data(group_JS,group_level_entropy_results)

plot_entropy_JSDist_corr(entropy_JS_corr_data,fname='entropy_JS_corr_data_multiclass.html')