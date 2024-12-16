import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
import plotly.express as px
import pandas as pd

from src.analysis.data_processing import (
    concat_colnames_nonzero,
    get_demographics_and_labels,
    get_demographics_and_llm_labels,
    get_wave_demographics,
    labels_16,
    wave_dates,
)
from src.analysis.metrics import (
    calculate_pmf_by_groups,
    calculate_pmf_population,
    get_MI_from_dataset,
    get_js_dist_by_groups,
    get_js_dist_population,
)
from src.paths import RESULTS_DIR
from scipy.spatial import distance
from scipy.stats import entropy


def visualize_cramer_waveExperiment_1x6(cramer_results, fname="cramer_waveExperiment_1x6.html"):
    """
    visualize in 1x6 , for the digital use
    """
    df=cramer_results
    df["index"] = df["index"].astype("category")
    df["x"] = df["index"].cat.codes * 0.3 + df["wave_id"] * 0.025

    xtick_position = [
        df[df["index"] == x_val].x.median() for x_val in df["index"].unique()
    ]
    x_mapping = {
        code: label for code, label in zip(xtick_position, df["index"].unique())
    }
    color_map = {
        "survey": "blue",  # Change 'source_1' to your actual source names
        "llm": "orange",  # Change 'source_2' to your actual source names
    }

    fig = go.Figure()

    for source in df["source"].unique():
        filtered_df = df[df["source"] == source]

        fig.add_trace(
            go.Scatter(
                x=filtered_df["x"],
                y=filtered_df["Cramers' V"],
                mode="markers+text",
                text=filtered_df["wave_id"],
                textposition="top center",
                name=source,
                marker=dict(size=10, color=color_map.get(source, "gray")),
                hoverinfo="text",
            )
        )

        for index_val in filtered_df["index"].unique():
            index_points = filtered_df[filtered_df["index"] == index_val]
            if len(index_points) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=index_points["x"],
                        y=index_points["Cramers' V"],
                        mode="lines",
                        line=dict(width=2, color=color_map.get(source, "gray")),
                        name=f"{source} - {index_val} Connection",
                        showlegend=False,
                    )
                )

    # Customizing the layout
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Cramer's V",
        title="Cramers V for Prompt Features",
        xaxis=dict(
            tickmode="array",
            tickvals=list(x_mapping.keys()),
            ticktext=list(x_mapping.values()),
        ),
        yaxis=dict(range=[0, df["Cramers' V"].max() + 0.01], tick0=0, dtick=0.01),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            title="Source",
            font=dict(size=14),
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0)",
            bordercolor="black",
            borderwidth=1,
        ),
    )
    fig.update_yaxes(gridcolor="lightgray", gridwidth=0.5, griddash="dash")
    fig.write_html(fname)
    return fig


def visualize_cramer_waveExperiment_2x3(cramer_results,fname='cramer_waveExperiment_2x3.html'):
    """
    visualize in 2x3 , for the printing
    """
    df=cramer_results
    df["index"] = df["index"].astype("category")
    df["x"] = df["index"].cat.codes * 0.3 + df["wave_id"] * 0.025

    unique_indices = df["index"].unique()
    num_cols = 3
    num_rows = (len(unique_indices) + num_cols - 1) // num_cols  # Calculate rows needed

    # Create subplots
    fig = make_subplots(rows=num_rows, cols=num_cols, shared_yaxes=True)

    color_map = {"survey": "blue", "llm": "orange"}

    # Add traces for each index
    for i, index_val in enumerate(unique_indices):
        row = i // num_cols + 1
        col = i % num_cols + 1
        for source in df["source"].unique():
            filtered_df = df[df["source"] == source]
            index_points = filtered_df[filtered_df["index"] == index_val]

            # Add scatter points
            fig.add_trace(
                go.Scatter(
                    x=index_points["x"],
                    y=index_points["Cramers' V"],
                    mode="markers+text",
                    text=index_points["Cramers' V"].round(3),
                    textposition="top center",
                    marker=dict(size=10, color=color_map[source]),
                    hoverinfo="text",
                    showlegend=False,  # No legend for individual traces
                ),
                row=row,
                col=col,
            )

            # Add lines if there are multiple points
            if len(index_points) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=index_points["x"],
                        y=index_points["Cramers' V"],
                        mode="lines",
                        line=dict(width=2, color=color_map[source]),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        fig.add_annotation(
            text=index_val,
            xref="paper",
            yref="paper",
            x=(col - 1) / num_cols + 1 / (2 * num_cols),  # Center in the column
            y=0.5 - ((row - 1) / num_rows - 0.05),  # Adjusted for better placement
            showarrow=False,
            font=dict(size=12),  # Adjust font size if needed
        )

    # Customizing the layout
    fig.update_layout(
        title_text="Cramers V for Prompt Features",
        title_x=0.5,  # Center the title
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Hide x-axis labels
    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            fig.update_xaxes(showticklabels=False, row=row, col=col)

    # Set y-axis title for each subplot
    for row in range(1, num_rows + 1):
        fig.update_yaxes(title_text="Cramer's V", row=row, col=1)

    fig.write_html(fname)
    return fig


def get_JS_group_plot_waveExperiment(survey_population_df, fname="JS_group_plot.html"):
    unique_groups = survey_population_df["social_group"].unique()
    colors = px.colors.qualitative.Set2
    group_colors = {
        category: colors[i % len(colors)] for i, category in enumerate(unique_groups)
    }

    survey_population_df = survey_population_df.sort_values(
        by=["wave", "social_group_category"]
    )

    fig = go.Figure()

    # Get unique categories in the order they appear on the x-axis
    categories = survey_population_df["social_group_category"].unique()

    for category in categories:
        subset = survey_population_df[
            survey_population_df["social_group_category"] == category
        ]
        social_groups = subset["social_group"].unique()

        for social_group in social_groups:
            group_subset = subset[subset["social_group"] == social_group]

            # Split the data into two parts: below and above 0.6
            below_threshold = group_subset[group_subset["js"] <= 0.6]
            above_threshold = group_subset[group_subset["js"] > 0.6]

            # Add the main line (below or equal to 0.6)
            fig.add_trace(
                go.Scatter(
                    x=[group_subset["social_group_category"], group_subset["wave"]],
                    y=group_subset["js"].clip(upper=0.6),
                    mode="lines+markers",
                    marker=dict(color=group_colors[social_group]),
                    line=dict(color=group_colors[social_group]),
                    text=group_subset["social_group"],
                    name=social_group,
                    legendgroup=category,
                    legendgrouptitle_text=category,
                    orientation="h"
                )
            )

            if not above_threshold.empty:
                # Find the first occurrence of a point above threshold
                for i in range(0, above_threshold.shape[0]):
                    point = above_threshold.iloc[i]
                    fig.add_trace(
                        go.Scatter(
                            x=[[point["social_group_category"]], [point["wave"]]],
                            y=[0.6],  # Set y to 0.6 for the point
                            mode="markers+text",
                            marker=dict(
                                color=group_colors[social_group],
                                symbol="triangle-up",
                                size=13,
                            ),
                            textposition="top center",
                            name=social_group,
                            showlegend=False,
                        )
                    )

    fig.update_layout(
        title="",#"JS Distance between Survey Answers and LLM Answers Over Time with Social Group Categories",
        xaxis_title="Survey Waves",
        yaxis_title="JS Distance",
        yaxis_range=[survey_population_df["js"].min() - 0.02, 0.62],
        margin=dict(b=100, t=100, r=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.7,  # Adjust this value to add more vertical space
            xanchor="center",
            x=0.5,
            title="",
            groupclick="toggleitem",
            itemsizing="constant",
            traceorder="grouped",

        ),
    )

    # Add annotation for values exceeding 0.6
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.02,
        y=1,
        text="▲ Values > 0.6",
        showarrow=False,
        font=dict(size=10),
        align="left",
        xanchor="left",
        yanchor="top",
    )

    fig.write_html(fname)



def plot_waveExperiment_population_comparison(
    df1,
    df2,
    title1="Survey Population",
    title2="LLM Population",
    output_file="waveExperiment_population_comparison.html",
):
    """
    compare categorey percentages over time
    """
    # Transpose DataFrames to get timestamps as rows and categories as columns
    df1 = df1.T
    df2 = df2.T

    # Convert DataFrame index to wave dates
    df1.index = df1.index.map(lambda x: wave_dates.get(x, x))
    df2.index = df2.index.map(lambda x: wave_dates.get(x, x))
    # Determine all unique categories from both datasets
    all_categories = sorted(set(df1.columns).union(set(df2.columns)))

    # Sort categories by the highest value in df1 in descending order
    sorted_categories = sorted(
        all_categories,
        key=lambda category: (
            df1[category].max() if category in df1.columns else float("-inf")
        ),
        reverse=True,
    )

    # Create subplots with 1 row for each category and 1 column
    fig = make_subplots(
        rows=len(sorted_categories),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[
            f"{i+1}. {category}" for i, category in enumerate(sorted_categories)
        ],
    )

    # Add traces for df1 and df2 for each category
    for i, category in enumerate(sorted_categories):
        if category in df1.columns:
            fig.add_trace(
                go.Scatter(
                    x=df1.index,
                    y=df1[category] * 100,  # Convert to percentage
                    mode="lines+markers",
                    name=f"{title1} - {category}",
                    hovertemplate="%{y:.2f}%",
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )

        if category in df2.columns:
            fig.add_trace(
                go.Scatter(
                    x=df2.index,
                    y=df2[category] * 100,  # Convert to percentage
                    mode="lines+markers",
                    name=f"{title2} - {category}",
                    hovertemplate="%{y:.2f}%",  # Only show percentage
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )

    ticktexts = [f"wave {k}:<br> {v}" for k, v in wave_dates.items()]
    for i in range(1, len(sorted_categories) + 1):
        fig.update_yaxes(title_text="Percentage (%)", tickformat=".2f", row=i, col=1)
        fig.update_xaxes(
            title_text="Survey Wave",
            tickvals=list(wave_dates.values()),
            ticktext=ticktexts,
            row=i,
            col=1,
        )
        fig.layout[f"xaxis{i}"]["showticklabels"] = True

    # Update overall layout
    fig.update_layout(
        height=300 * len(sorted_categories),
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False,
        # xaxis_showticklabels=True,
    )

    # Save the plot as an HTML file
    fig.write_html(output_file)

    # Return the plot figure
    return fig


def plot_entropy_JSDist_corr(entropy_JS_corr_data,fname='entropy_JS_corr_data.html'):
    k=entropy_JS_corr_data
    # Define marker symbols
    symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "star",
        "circle-cross",
        "cross-thin",
        "diamond-tall",
        "square-cross",
        "hexagon",
        "asterisk",
    ]
    colors = px.colors.qualitative.Set2  # Fixed colors for each subplot

    # Assign unique symbols within each social_group_category
    symbol_map = {}
    for category in k["social_group_category"].unique():
        symbol_map[category] = symbols[
            : k[k["social_group_category"] == category].shape[0]
        ]

    # Create subplots
    categories = k["social_group_category"].unique()
    num_categories = len(categories)
    rows = (num_categories + 2) // 3  # 2 columns layout

    fig = make_subplots(
        rows=rows,
        cols=3,
        subplot_titles=[cat for cat in categories],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    # Add scatter plots for each social_group_category
    for i, category in enumerate(categories):
        row = i // 3 + 1
        col = i % 3 + 1
        category_data = k[k["social_group_category"] == category]

        # Calculate linear regression for trendline
        slope, intercept, r_value, p_value, std_err = linregress(
            category_data["entropy"], category_data["js"]
        )
        r_squared = r_value**2  # Calculate R²

        # Add scatter trace
        for j in range(len(category_data)):
            position = "middle right"  # Default text position
            font_size = 12  # Default font size
            if category == "berufabschluss_clause":
                font_size = 10  # Reduced font size for specific category
            if j == len(category_data) - 1:  # Last dot
                position = "middle left"  # Adjust text position for last point

            fig.add_trace(
                go.Scatter(
                    x=[category_data["entropy"].iloc[j]],
                    y=[category_data["js"].iloc[j]],
                    mode="markers+text",
                    text=[category_data["text"].iloc[j]],
                    marker=dict(
                        size=18,
                        symbol=symbol_map[category][0],
                        color=colors[i % len(colors)],
                    ),  # Fixed color for each category
                    textposition=position,
                    textfont=dict(size=font_size),
                ),
                row=row,
                col=col,
            )

        # Add trendline
        x_range = [category_data["entropy"].min(), category_data["entropy"].max()]
        y_range = [intercept + slope * x for x in x_range]

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                mode="lines",
                name="Trendline",
                line=dict(color="gray", dash="dash"),  # Dashed line
            ),
            row=row,
            col=col,
        )

        # Update x and y axis labels
        fig.update_xaxes(title_text="Subgroup's Survey Entropy", row=row, col=col)
        fig.update_yaxes(title_text="JS Distance", row=row, col=col)

    # Update layout
    fig.update_layout(
        title_text="Correlation Scatter Plots by Social Group Category",
        showlegend=False,
    )

    # Show plot
    fig.write_html(fname)


def get_survey_to_survey_JS_distances(survey_population_df,fname='s2s_JS_dist.html'):
    rs=[]
    for wave_id in [17,18,19,20,21]:
        for col in survey_population_df:
            if (col<=wave_id):
                js=distance.jensenshannon(survey_population_df[col],survey_population_df[wave_id])
                r={'d1':col,
                'd2':wave_id,
                'js':js}
                rs.append(r)
    df=pd.DataFrame(rs)

    df['text']="wave" + df['d1'].astype(str)+ "<br>"+ df['d1'].map(wave_dates)
    df['d2']=df['d2'].astype(str)
    import plotly.graph_objects as go
    fig = go.Figure()

    # Add a trace for each category in d2
    for category in df['d2'].unique():
        category_df = df[df['d2'] == category]
        fig.add_trace(go.Scatter(
            x=category_df['d1'],
            y=category_df['js'],
            mode='lines+markers',
            name=str(category),
            text=category_df['text'],
        ))

    # Update layout
    fig.update_layout(
        title='',#'Survey to Survey JS Distances for Waves 19,20,21; Compared to Waves (10-19),(10-20),(10-21)',
        xaxis_title='Wave - Date',
        yaxis_title='JS',
        xaxis_tickvals=df['d1'],
        xaxis_ticktext=df['text']
    )

    # Save to HTML
    fig.write_html(fname)
    return fig



def entropy_by_social_group(group_level_entropy_results,fname='entropy_by_social_group.html'):
    '''
    averaged across waves/experiments, survey_entropy values and llm_entropy values 
    '''
    df= group_level_entropy_results.groupby(['social_group','source'])['entropy'].mean().reset_index()

    # Pivot the DataFrame
    df_pivot = df.pivot(index='social_group', columns='source', values='entropy').reset_index()

    # Create the scatter plot
    fig = px.scatter(df_pivot, x='survey', y='llm', text='social_group', 
                     title='Survey vs LLM Entropy by Social Group',
                     labels={'survey': 'Survey Entropy', 'llm': 'LLM Entropy'})

    # Add the y=x line
    fig.add_trace(go.Scatter(x=[min(df_pivot['survey']), max(df_pivot['survey'])], 
                             y=[min(df_pivot['survey']), max(df_pivot['survey'])], 
                             mode='lines', 
                             name='y=x', 
                             line=dict(color='Red', dash='dash')))

    # Add social group labels to each point
    fig.update_traces(textposition='top center')

    # Show plot
    fig.write_html(fname)


def get_labels_percentage_table(survey_population_df_multilabel2,llm_population_df_multilabel2):

    def color_cell(value, threshold=0):
        value=float(value)
        if value > 1:
            color = 'ForestGreen'
            return f"\\textcolor{{{color}}}{{value}}"
        elif value < -1 :
            color='red'
            return f"\\textcolor{{{color}}}{{value}}"
        else:
            color='black'
            return f"\\textcolor{{{color}}}{{value}}"


    a= survey_population_df_multilabel2.multiply(100).round(1)#.astype(str)
    b= llm_population_df_multilabel2.multiply(100).round(1)#.astype(str)
    colordf=(b-a).applymap(color_cell)
    for col in colordf.columns:
        colordf[col]=colordf[col].combine(b[col],lambda fmt,value: fmt.replace('value',str(value)))
    b=colordf#.combine(b[12],lambda fmt,value: fmt.format(value))
    
    a['src']='survey'
    b['src']='llm'
    a=a.astype(str)
    b.loc[:,'mean APE']= ( llm_population_df_multilabel2-survey_population_df_multilabel2).divide(survey_population_df_multilabel2).multiply(100).abs().mean(axis=1).round(2).astype(str)
    #b.loc[:,'mean PE']= ( llm_population_df_multilabel2-survey_population_df_multilabel2).divide(survey_population_df_multilabel2).multiply(100).mean(axis=1)

    c=pd.concat([a.set_index([a.index,'src']),b.set_index([b.index,'src'])]).sort_index()
    c['mean APE']= c['mean APE'].fillna('')
    #c['mean PE']= c['mean PE'].fillna('')

    c.loc['APE',:]=np.append(( llm_population_df_multilabel2-survey_population_df_multilabel2).round(2).multiply(100).abs().sum(axis=0).values , [''])
    return c #c#.to_latex('a.txt')
    
#get_labels_percentage_table(survey_population_df_multilabel2,llm_population_df_multilabel2).to_latex('multilabel_comp_waveExperiment.txt')
#get_labels_percentage_table(survey_population_df_multiclass1,llm_population_df_multiclass1)#.to_latex('multiclass_comp_waveExperiment.txt')


# get_survey_to_survey_JS_distances(survey_population_df)

# plot_entropy_JSDist_corr(k)
# fig = visualize_cramer_waveExperiment_1x6(cramer_results, "cramer_results.html")
# fig = visualize_cramer_waveExperiment_1x6(cramer_results2, "cramer_results2.html")
# fig = visualize_cramer_waveExperiment_2x3(cramer_results)
# fig = get_JS_group_plot_waveExperiment(group_JS_df)
# fig = plot_waveExperiment_population_comparison(survey_population_df, llm_population_df)
