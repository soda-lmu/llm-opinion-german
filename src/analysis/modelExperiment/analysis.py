from src.analysis.ablationExperiment.plot import get_ablation_JS_plot
from src.analysis.experiment_utils import get_JS_experiment, get_experiment_entropy
from src.analysis.metrics import get_cramerV, get_population_level_ape_results
from src.analysis.modelExperiment.plot import get_modelExperiment_pmf_comparison
from src.analysis.modelExperiment.utils import get_modelExperiment_data, get_textual_stats
from src.analysis.data_processing import labels_16
(
survey_labels_dict1,
llm_labels_dict1,
# multilabel
survey_population_df_multilabel1,  # df
llm_population_df_multilabel1,  # df
survey_group_pmf_multilabel1,  # dict of dfs
llm_group_pmf_multilabel1,  # dict of dfs
# multiclass
survey_population_df_multiclass1,  # df
llm_population_df_multiclass1,  # df
survey_group_pmf_multiclass1,  # dict of dfs
llm_group_pmf_multiclass1,  # dict of dfs
)=get_modelExperiment_data(save=False)
print('data loaded')


population_JS, group_JS = get_JS_experiment(
    survey_population_df_multilabel1, llm_population_df_multilabel1, survey_group_pmf_multilabel1, llm_group_pmf_multilabel1
)
population_level_entropy_results, group_level_entropy_results = (
    get_experiment_entropy(
        survey_population_df_multilabel1, llm_population_df_multilabel1, survey_group_pmf_multilabel1, llm_group_pmf_multilabel1
    )
)


js_population_fig= get_ablation_JS_plot(population_JS,save=True,fname='js_population_fig_multilabel.html')
js_population_fig.write_html('js_population_fig_multilabel.html')

population_JS.to_csv('population_JS_multilabel.csv')
population_level_entropy_results.to_csv('population_level_entropy_results_multilabel.csv')

ape_results= get_population_level_ape_results(survey_population_df_multiclass1,llm_population_df_multiclass1,survey_population_df_multilabel1,llm_population_df_multilabel1,save=True,experiment_type='modelExperiment',file_name='ape_results.csv')
print('ape results saved')
fig= get_modelExperiment_pmf_comparison(survey_population_df=survey_population_df_multilabel1,llm_population_df= llm_population_df_multilabel1, fname= 'pmf_comparison_multilabel_1.html', save=True)
fig= get_modelExperiment_pmf_comparison(survey_population_df=survey_population_df_multiclass1,llm_population_df= llm_population_df_multiclass1, fname= 'pmf_comparison_multiclass_1.html', save=True)
print('pmf charts saved')
get_textual_stats(llm_labels_dict1,survey_labels_dict1).to_csv('textual_stats.csv')
print('textual_stats csv saved')
