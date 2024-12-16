
from src.analysis.ablationExperiment.utils import get_ablation_cramer_table, get_ablationExperiment_data

from src.analysis.metrics import  get_entropy_JS_corr_data,get_cramerV, get_cramerV_multiclass, get_population_level_ape_results
from src.analysis.ablationExperiment.plot import get_ablation_JS_plot
from src.analysis.experiment_utils import get_experiment_entropy, get_JS_experiment, get_entropy_JS_corr_data
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
)=get_ablationExperiment_data()


method = "multilabel"
survey_population_df, llm_population_df, survey_group_pmf, llm_group_pmf = (
    survey_population_df_multilabel,
    llm_population_df_multilabel,
    survey_group_pmf_multilabel,
    llm_group_pmf_multilabel,
)

#MI_results_waveExperiment = get_MI_experiment(survey_labels_dict, llm_labels_dict)
population_JS, group_JS = get_JS_experiment(
    survey_population_df_multilabel, llm_population_df, survey_group_pmf, llm_group_pmf
)
population_level_entropy_results, group_level_entropy_results = (
    get_experiment_entropy(
        survey_population_df_multilabel, llm_population_df, survey_group_pmf, llm_group_pmf
    )
)


cramer_results = get_cramerV(survey_labels_dict, llm_labels_dict)
get_ablation_cramer_table(cramer_results,save=True,fname='cramer_results_multiclass.csv')


js_population_fig= get_ablation_JS_plot(population_JS,save=True,fname='js_population_fig_multilabel.html')
js_population_fig.write_html('js_population_fig_multilabel.html')
ape_results= get_population_level_ape_results(survey_population_df,llm_population_df,survey_group_pmf,llm_group_pmf,save=True,experiment_type='ablationExperiment',file_name='ape_results_multilabel.csv')



##############################################################################################
method="multiclass"
survey_population_df, llm_population_df, survey_group_pmf, llm_group_pmf = (
    survey_population_df_multiclass,  # df
    llm_population_df_multiclass,  # df
    survey_group_pmf_multiclass,  # dict of dfs
    llm_group_pmf_multiclass,  # dict of dfs
)

#MI_results_waveExperiment = get_MI_experiment(survey_labels_dict, llm_labels_dict)
population_JS, group_JS = get_JS_experiment(
    survey_population_df_multiclass, llm_population_df_multiclass, survey_group_pmf_multiclass, llm_group_pmf_multiclass
)
population_level_entropy_results, group_level_entropy_results = (
    get_experiment_entropy(
        survey_population_df_multiclass, llm_population_df_multiclass, survey_group_pmf_multiclass, llm_group_pmf_multiclass
    )
)
cramer_results_multiclass = get_cramerV_multiclass(
    survey_labels_dict, llm_labels_dict
)
entropy_JS_corr_data=get_entropy_JS_corr_data(group_JS,group_level_entropy_results)

js_population_fig= get_ablation_JS_plot(population_JS,save=True,fname='js_population_fig_multiclass.html')
js_population_fig.write_html('js_population_fig_multiclass.html')

get_ablation_cramer_table(cramer_results_multiclass,save=True,fname='cramer_results_multiclass.csv')
####################

ape_results= get_population_level_ape_results(survey_population_df,llm_population_df,survey_group_pmf,llm_group_pmf,save=True,experiment_type='ablationExperiment',file_name='ape_results_multiclass.csv')

