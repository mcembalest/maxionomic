from itertools import islice
import numpy as np
import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from typing import Dict

from nomic.atlas import AtlasDataset
from latentsae import Sae

import dash
from dash import html, dcc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# load models
sae_model = Sae.load_from_hub("enjalot/sae-nomic-text-v1.5-FineWeb-edu-100BT", "64_32")
emb_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
device = "mps"
sae_model = sae_model.to(device)
emb_model = emb_model.to(device)

# load sae features
loaded_features = pd.read_parquet("features.parquet").to_dict(orient='records')

# Load data from file if it exists. Otherwise, download from Nomic Atlas.
data_file = 'yc_data.pkl'
if os.path.exists(data_file):
    with open(data_file, 'rb') as f:
        yc_df, yc_embeddings, yc_projected_embeddings, years = pickle.load(f)
else:
    datamap = AtlasDataset('nomic/y-combinator').maps[0]
    yc_df = datamap.data.df
    yc_embeddings = datamap.embeddings.latent
    yc_projected_embeddings = datamap.embeddings.projected
    yc_projected_embeddings['year'] = yc_df.Year

    selection_idx = yc_df[(yc_df.Year > 0) & (yc_df.oneliner_then_tags != "null")].index.values
    yc_df = yc_df.loc[selection_idx]
    yc_projected_embeddings = yc_projected_embeddings.loc[selection_idx]
    years = sorted(yc_projected_embeddings['year'].unique())
    with open(data_file, 'wb') as f:
        pickle.dump((yc_df, yc_embeddings, yc_projected_embeddings, years), f)


def aggregate_encoder_output(encoder_output, k: int = 5) -> Dict[int, float]:
    total_activations = {}
    for idx, act in zip(encoder_output.top_indices.cpu().flatten(), encoder_output.top_acts.cpu().flatten()):
        idx_int = idx.item()
        if idx_int in total_activations:
            total_activations[idx_int] += act.item()
        else:
            total_activations[idx_int] = act.item()
    sorted_activations = dict(sorted(total_activations.items(), key=lambda item: item[1], reverse=True))
    return sorted_activations

def summarize_encoder_output(sorted_activations, k=5):
    return [loaded_features[idx]['label'] for idx in list(islice(sorted_activations, k))]

# Precompute bar chart data for each year or load from file
bar_chart_data_file = 'bar_chart_data.pkl'
if os.path.exists(bar_chart_data_file):
    with open(bar_chart_data_file, 'rb') as f:
        bar_chart_data = pickle.load(f)
else:
    bar_chart_data = {}
    for year in years:
        s = yc_df[yc_df.Year == year].oneliner_then_tags.values
        text_embeddings = emb_model.encode(s, convert_to_tensor=True, normalize_embeddings=True)
        top_activated_features_sae_output = sae_model.encode(text_embeddings)
        top_sae_features_hist = aggregate_encoder_output(top_activated_features_sae_output)
        idx = list(top_sae_features_hist.keys())[:10]
        names = [f'{i}: {loaded_features[i]["label"]}' for i in idx]
        vals = [top_sae_features_hist[i] for i in idx]
        bar_chart_data[year] = {'names': names, 'vals': vals}
    with open(bar_chart_data_file, 'wb') as f:
        pickle.dump(bar_chart_data, f)


def create_figure(selected_year):
    fig = make_subplots(
        rows=2, 
        cols=2, 
        subplot_titles=(
            f"Y Combinator Startups {selected_year}", 
            f"Feature Activation",
            "Year Selection"
        ),
        column_widths=[0.7, 0.3],
        row_heights=[0.8, 0.2],
        vertical_spacing=0.1,
        specs=[
            [{"colspan": 1}, {"colspan": 1}],
            [{"colspan": 2}, None]
        ]
    )
    
    # Main scatter plot
    fig.add_trace(go.Scatter(
        x=yc_projected_embeddings['x'],
        y=-yc_projected_embeddings['y'],
        mode='markers',
        marker=dict(
            size=6,
            color=['red' if year == selected_year else 'lightgrey' for year in yc_projected_embeddings['year']],
            opacity=0.7
        ),
        text=[
            f'{row.Company} {row.Batch} {row.Status}<br><br>{row.oneliner_then_tags}' 
            for _, row in yc_df.iterrows()
        ],
        hoverinfo='text'
    ), row=1, col=1)

    # Bar chart
    fig.add_trace(go.Bar(
        y=bar_chart_data[selected_year]['names'][::-1], 
        x=bar_chart_data[selected_year]['vals'][::-1],
        orientation='h',
        name='SAE Features',
        text=[name[:75] + '...' if len(name) > 75 else name for name in reversed(bar_chart_data[selected_year]['names'])],
        textposition='inside',
        insidetextanchor='start',
        textfont=dict(color='white'),
        hoverinfo='text'
    ), row=1, col=2)

    # Year selection scatter plot
    fig.add_trace(go.Scatter(
        x=years,
        y=[1] * len(years),
        mode='markers',
        marker=dict(
            size=10,
            color=['red' if year == selected_year else 'blue' for year in years],
            opacity=0.8
        ),
        text=[str(year) for year in years],
        hoverinfo='text'
    ), row=2, col=1)

    fig.update_layout(
        height=800,
        showlegend=False,
        hovermode='closest'
    )
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_yaxes(title_text="SAE Features", tickangle=90, row=1, col=2, tickfont=dict(size=8))
    fig.update_xaxes(title_text="Activation", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    

    return fig

# Update the app layout
app.layout = html.Div([
    dcc.Graph(id='main-graph', style={'height': '800px'}),
    dcc.Store(id='selected-year', data=min(years))
])

@app.callback(
    Output('main-graph', 'figure'),
    Output('selected-year', 'data'),
    Input('main-graph', 'hoverData'),
    Input('selected-year', 'data')
)
def update_graph(hover_data, current_year):
    ctx = dash.callback_context
    if not ctx.triggered:
        return create_figure(current_year), current_year

    input_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if input_id == 'main-graph':
        if hover_data and 'points' in hover_data:
            point = hover_data['points'][0]
            if 'x' in point and point['curveNumber'] == 2:  # Year selection scatter plot
                selected_year = int(point['x'])
                return create_figure(selected_year), selected_year

    return create_figure(current_year), current_year


if __name__ == '__main__':
    app.run_server(debug=False)