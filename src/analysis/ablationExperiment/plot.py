from src.analysis.ablationExperiment.utils import ablation_mapped_dict
import pandas as pd
import plotly.express as px


def get_ablation_JS_plot(population_JS, save, fname='ablation_js.html', bar_width=0.8, text_size=12):
    def filter_index(row):
        a = row['index'][0]
        exp_values = ablation_mapped_dict[a]
        wave_id = row['wave_id'][0]
        if (wave_id in exp_values) and (wave_id != 'Llama2_base') and ('without_' not in wave_id):
            return True
        else:
            return False

    def get_experiment_type(row):
        a = row['exp']
        if '1VAR' in a:
            return 'one variable'
        elif 'all' in a:
            return 'all variables'
        elif 'without' in a:
            return 'all except one variable'
        elif 'base' in a:
            return 'no demographics'
        else:
            return None

    def get_experiment_str(row):
        a = row['exp']
        if '1VAR' in a:
            var = a.strip('1VAR_')
            return f'{var} only'
        elif 'all' in a:
            return 'all variables'
        elif 'without' in a:
            var = a.strip('without_')
            return f'all except {var}'
        elif 'base' in a:
            return 'no demographics'
        else:
            return None

    population_JS = population_JS.T.reset_index().rename({'index': 'exp', 0: 'js'}, axis=1)
    population_JS['exp_type'] = population_JS.apply(get_experiment_type, axis=1)
    population_JS['exp_str'] = population_JS.apply(get_experiment_str, axis=1)
    categories = ['no demographics', 'age only', 'berufabschluss only', 'eastwest only', 'gender only',
                  'party only', 'schulabschluss only',
                  'all except age', 'all except berufabschluss',
                  'all except eastwes', 'all except gender', 'all except party',
                  'all except schulabschluss', 'all variables']

    population_JS['exp_str'] = pd.Categorical(population_JS['exp_str'], categories=categories, ordered=True)
    population_JS = population_JS.sort_values(by='exp_str')
    ordered_categories = ['no demographics', 'one variable', 'all variables', 'all except one variable']
    ablation_js_population = population_JS

    fig = px.bar(
        ablation_js_population,
        x='exp',
        y='js',
        color='exp_type',
        category_orders={'exp_str': categories},
    )

    # Adjust bar width and text size
    fig.update_traces(marker=dict(line=dict(width=bar_width)))
    fig.update_layout(font=dict(size=text_size))

    max_level = ablation_js_population['js'].max()
    min_level = ablation_js_population['js'].min()
    fig.add_shape(type="line", x0=-0.5, y0=max_level, x1=len(ablation_js_population['exp']) - 0.5, y1=max_level,
                  line=dict(color="gray", width=2, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, y0=min_level, x1=len(ablation_js_population['exp']) - 0.5, y1=min_level,
                  line=dict(color="gray", width=2, dash="dash"))

    # Update x-axis tick labels
    fig.update_layout(
        xaxis_title='Experiment Type',
        yaxis_title='JS',
        xaxis_tickvals=ablation_js_population['exp'],
        xaxis_ticktext=ablation_js_population['exp']  # Display label for x-axis
    )
    fig.update_yaxes(dtick=0.05)

    fig.update_layout(
        xaxis_title='Ablation Experiment : Experiment Types - JS Distance',
        yaxis_title='JS'
    )
    fig.update_layout(
        xaxis_title='Experiment Type',
        yaxis_title='JS',
        xaxis_tickvals=list(range(len(ablation_js_population))),
        xaxis_ticktext=ablation_js_population['exp_str']
    )
    if save:
        fig.write_html(fname)
    return fig

#js_population_fig= get_ablation_JS_plot(population_JS,save=True,fname='js_population_fig_multilabel.html', bar_width=0.1, text_size=18)



