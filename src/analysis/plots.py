from src.analysis.data_processing import coarse_translation, wave_dates

def get_label_distr_time_series_plot_data(survey_labels_dict2):
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    # Prepare lists to hold data for plotting
    plot_data = []

    for key, df in survey_labels_dict2.items():
        # Get value counts for the current DataFrame
        counts = df['highest_prob_label'].value_counts(1).reset_index()
        counts.columns = ['Answer Labels', 'freq']  # Renamed column

        # Add a column for time
        counts['time'] = wave_dates[key]

        # Append to list
        plot_data.append(counts)

    # Combine all data into a single DataFrame
    plot_df = pd.concat(plot_data, ignore_index=True)
    plot_df['Answer Labels'] = plot_df['Answer Labels'].map(coarse_translation)  # Updated column name
    return plot_df
    # Calculate the average frequency for each label
    avg_freq = plot_df.groupby('Answer Labels')['freq'].mean().reset_index()
    avg_freq = avg_freq.sort_values(by='freq', ascending=False)

    # Get the top 5 labels
    top_5_lines = avg_freq.head(5)['Answer Labels']

    # Choose a distinct color palette
    color_palette = px.colors.qualitative.Plotly

    # Plotting
    fig = px.line(plot_df, x='time', y='freq', color='Answer Labels', markers=True,
                  labels={'freq': 'Number of Occurrences', 'time': 'Date', 'Answer Labels': 'Answer Labels'},
                 color_discrete_sequence=color_palette)

    fig.update_layout(
        xaxis_title='GLES Survey Dates',
        yaxis_title='Frequency',
        title_x=0.5,  # Center the title

        # Make X and Y axis values bold by using a bold font family
        xaxis=dict(
            tickfont=dict(size=17, family="Arial Black", color="black"),
            tickangle=-45
        ),
        yaxis=dict(
            tickfont=dict(size=17, family="Arial Black", color="black")
        ),

        font=dict(size=17, family="Arial Black", color="black"),  # General text bold
        width=12 * 80,  # Increased width (12 inches)
        height=11.69 * 80,  

        # Modify legend to bold items and title with new line
        legend=dict(
            title=dict(text="", font=dict(size=17, family="Arial Black")),  # Bold legend title
            font=dict(size=17, family="Arial Black"),  # Bold legend items
            orientation='h',  # Horizontal layout for legend items
            yanchor='top',
            y=-0.3,  # Position legend further down to avoid overlap with x-axis title
            xanchor='center',
            x=0.5,  # Center the legend horizontally
            itemsizing='constant',
            itemwidth=60  # Set a width for each item to control spacing
        ),

        # Remove background and adjust margins to center the figure properly
        plot_bgcolor='white',  # Background color of the plot area
        paper_bgcolor='white',  # Background color of the entire paper area
        margin=dict(l=80, r=80, t=80, b=200)  # Adjust margins; increase bottom margin for legend
    )

    # Add text labels
    fig.update_traces(text=plot_df['freq'], textposition='top center')

    # Add annotations with arrows for the peak points of the top 5 lines
    annotations = []
    for i, label in enumerate(top_5_lines):
        # Find the corresponding line trace
        trace_color = None
        for trace in fig.data:
            if trace.name == label:
                trace_color = trace.line.color
                break

        # Get the peak point for the annotation
        line_data = plot_df[plot_df['Answer Labels'] == label]
        peak_point = line_data.loc[line_data['freq'].idxmax()]
        x_val = peak_point['time']
        y_val = peak_point['freq']

        # Create the annotation with the matching color
        annotations.append(dict(
            x=x_val,
            y=y_val,
            text=f"{label}",
            showarrow=True,
            arrowhead=2,
            ax=120,
            ay=-30 - i * 20,
            font=dict(size=17, family="Arial Black", color=trace_color)  # Use the line's color
        ))

    fig.update_layout(annotations=annotations)

    return fig
    # fig.write_html('label_distr_time_series_plot.html')
