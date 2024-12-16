def get_modelExperiment_pmf_comparison(llm_population_df,survey_population_df,fname='pmf_comparison_1.html',save=False):
    df = llm_population_df.copy()
    df['wave 12']=survey_population_df.iloc[:,0]
    df.columns=['gemma-7b-it', 'lama-2-13b-chat-hf',
           'mistralai-Mixtral-8x7B-Instruct', 'wave 12']
    df= df.apply(lambda x: (x*100).round(1) ) #['wave 12']
    coarse_translation_formatted = {
        "Politische Strukturen und Prozesse": "Political System <br> and Processes",
        "Sozialpolitik": "Social <br> Policy",
        "Gesundheitspolitik": "Health <br> Policy",
        "Familien- und Gleichstellungspolitik": "Family and <br> Gender Equality <br> Policy",
        "Bildungspolitik": "Education <br> Policy",
        "Umweltpolitik": "Environmental <br> Policy",
        "Wirtschaftspolitik": "Economic <br> Policy",
        "Sicherheits": "Security",
        "Außenpolitik": "Foreign <br> Policy",
        "Medien und Kommunikation": "Media and <br> Communication",
        "Sonstiges": "Others",
        "Migration und Integration": "Migration and <br> Integration",
        "Ostdeutschland": "East <br> Germany",
        "keine Angabe": "Not <br> specified",
        "weiß nich": "Do not know",
        "LLM refusal": "LLM refusal",
        "Werte, politische Kultur und Gesellschaftskritik": "Values,<br> political culture<br> and general <br> social criticism"
    }

    df.index=df.index.map(coarse_translation_formatted)
    import plotly.graph_objects as go

    def plot_comparison_chart(llm_population_df, title, output_file_path, save=False, width=1):
        figs = []
        for col in llm_population_df.columns:
            data = llm_population_df[col]
            figs.append(go.Bar(
                x=data.index, 
                y=data.values, 
                name=col,
                width=width,
                 text=data.values,         
                textposition='outside', )
            )

        # Combine the traces in a single figure
        fig = go.Figure(data=figs)

        fig.update_layout(
            title=title,
            yaxis_title='Percentage',
            barmode='group',  
            bargap=0.30,  
            bargroupgap=0.35,  
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
            ),     
            font=dict(size=15),
            plot_bgcolor='rgba(0, 0, 0, 0)',
        )

        def split_label(label):
            if isinstance(label, str) and len(label) > 10:
                middle = len(label) // 2
                space_pos = label.rfind(' ', 0, middle)
                comma_pos = label.rfind(',', 0, middle)

                split_pos = max(space_pos, comma_pos) if max(space_pos, comma_pos) != -1 else middle

                return f"{label[:split_pos+1]}<br>{label[split_pos+1:]}"
            return label

        tickvals = llm_population_df.index
        ticktext = [split_label(x) for x in tickvals]

        # Generate positions for separator lines based on the index of categorical values
        separator_positions = [i + 0.5 for i in range(len(tickvals) - 1)]

        fig.update_xaxes(
            ticktext=ticktext,
            tickvals=tickvals,
            tickfont=dict(color='black'),
            tickangle=0,  # Set tick angle to 0 to make text horizontal
            showgrid=False,
            ticks='outside',
            ticklen=10,  # Length of the ticks
            tickwidth=2,  # Width of the ticks
            tickcolor='white'
        )

        fig.update_yaxes(tickfont=dict(color='black'))

        # Add separator lines using shapes
        shapes = []
        for pos in separator_positions:
            shapes.append(dict(
                type='line',
                x0=pos,
                x1=pos,
                y0=0,
                y1=-0.025,
                xref='x',
                yref='paper',
                line=dict(color='black', width=2)
            ))

        fig.update_layout(shapes=shapes)

        if save and output_file_path.endswith(".html"):    
            # Save the figure as an HTML file
            fig.write_html(output_file_path)
        elif save and output_file_path.endswith(".png"):
            fig.write_image(output_file_path)
        else:
            fig.write_image(output_file_path, engine="kaleido")

        return fig
    fig=plot_comparison_chart(llm_population_df=df, title = "", output_file_path= fname, save=True, width=0.2)
    return fig
