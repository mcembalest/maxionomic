import json
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
import argparse
import matplotlib.font_manager as fm
fm.fontManager.addfont("/Library/Fonts/Inter-VariableFont_opsz,wght.ttf")
fm.fontManager.addfont("/Library/Fonts/CrimsonPro-VariableFont_wght.ttf")

parser = argparse.ArgumentParser(description='Plot performance vs cost for embedding models')
parser.add_argument('--dataset', type=str, default='msmarco',
                   help='Dataset to plot (default: msmarco)')
args = parser.parse_args()

COLORS = ['#0D4A34', '#E5B300', '#0D4A34', '#179266']
EC2_COST_PER_SECOND = 5.67 / 3600  # $5.67 per hour (G5.12xlarge cost) converted to per second
VOYAGE_RERANK_COST_PER_MILLION_TOKENS = 0.05

plt.rcParams.update({
    'font.family': 'Inter',
    'font.size': 10,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.facecolor': '#FEFBF6',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#0D4A34',
    'axes.labelcolor': '#0D4A34',
    'xtick.color': '#0D4A34',
    'ytick.color': '#0D4A34',
    'grid.color': '#0D4A34',
    'grid.alpha': 0.1,
    'lines.color': '#0D4A34',
    'patch.edgecolor': '#0D4A34',
    'boxplot.boxprops.color': '#0D4A34',
    'axes.prop_cycle': plt.cycler('color', COLORS),
})

with open('performance_stats.json', 'r') as f:
    perf_data = json.load(f)['data']
dataset_data = [d for d in perf_data if d['dataset'] == args.dataset]
latency_stats = {}
reranking_stats = {}

# Load reranking stats
for filename in os.listdir('reranking_stats'):
    if args.dataset in filename:
        with open(os.path.join('reranking_stats', filename), 'r') as f:
            data = json.load(f)
            model_name = data['model'].replace('nomic-ai/', '').replace('voyageai/', '')
            # Store the actual cost data for reranking
            reranking_stats[model_name] = {
                'total_tokens_reranked': data['metrics']['total_tokens_reranked'],
                'total_reranking_requests': data['metrics']['total_reranking_requests'],
                'queries_evaluated': data['metrics']['queries_evaluated']
            }

for filename in os.listdir('latency_stats'):
    if args.dataset in filename:
        with open(os.path.join('latency_stats', filename), 'r') as f:
            data = json.load(f)
            model_name = data['model'].replace('nomic-ai/', '').replace('voyageai/', '')
            # Store tokens per second and total tokens from latency stats
            latency_stats[model_name] = {
                'tokens_per_second': data['tokens_per_second'],
                'total_tokens': data['total_tokens']
            }

plot_data = []
for entry in dataset_data:
    model_name = entry['model'].replace('nomic-ai/', '').replace('voyageai/', '')
    if model_name in latency_stats:
        # Get tokens per second and total tokens from latency stats
        tokens_per_second = latency_stats[model_name]['tokens_per_second']
        total_tokens = latency_stats[model_name]['total_tokens']
        
        # Calculate time to process all tokens
        processing_time = total_tokens / tokens_per_second
        # Calculate embedding cost
        embedding_cost = processing_time * EC2_COST_PER_SECOND
        
        # Total cost starts with embedding cost
        total_cost = embedding_cost
        
        # Add reranking cost if applicable
        if entry['Reranking']:
            # Calculate actual reranking cost
            reranking_cost = reranking_stats[model_name]['total_tokens_reranked'] / 1_000_000 * VOYAGE_RERANK_COST_PER_MILLION_TOKENS
            total_cost += reranking_cost
        
        plot_data.append({
            'model': model_name,
            'MRR@10': entry['MRR@10'],
            'cost': total_cost,
            'Reranking': 'With Reranking' if entry['Reranking'] else 'Without Reranking'
        })

plt.figure(figsize=(10, 6))

def format_scientific(x, p):
    return f'${x:.4f}'

ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(format_scientific))
ax.grid(True, which='major', linestyle='--', alpha=0.2)
ax.grid(True, which='minor', linestyle=':', alpha=0.1)
ax.minorticks_on()
colors = {'nomic-embed-text-v1.5': COLORS[3], 'voyage-3-large': COLORS[1]}
markers = {'Without Reranking': 'o', 'With Reranking': '^'}
model_points = {}
for entry in plot_data:
    if entry['model'] not in model_points:
        model_points[entry['model']] = []
    model_points[entry['model']].append(entry)
for model, points in model_points.items():
    points.sort(key=lambda x: x['Reranking'])
    plt.plot(
        [p['cost'] for p in points],
        [p['MRR@10'] for p in points],
        color=colors[model],
        alpha=0.3,
        linestyle='--',
        zorder=1
    )

for entry in plot_data:
    x = entry['cost']
    y = entry['MRR@10']
    scatter = plt.scatter(
        x, y,
        c=colors[entry['model']],
        marker=markers[entry['Reranking']],
        s=150,
        label=f"{entry['model']} ({entry['Reranking']})",
        zorder=2
    )
    
    model_display = 'nomic-ai/nomic-embed-text-v1.5' if 'nomic' in entry['model'].lower() else 'voyageai/voyage-3-large'
    rerank_display = '\n+ voyageai/rerank-2' if entry['Reranking'] == 'With Reranking' else ''
    
    #####################################################
    # make sure annotations dont go outside the plot
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    x_rel = (x - x_min) / (x_max - x_min)
    y_rel = (y - y_min) / (y_max - y_min)
    
    # Adjust text position based on location in plot
    if x_rel > 0.8:  # Point is near right edge
        x_text = -10
        ha = 'right'
    else:  # Point is elsewhere
        x_text = 10
        ha = 'left'
    
    if y_rel > 0.8:  # Point is near top
        y_text = -10
        va = 'top'
    else:  # Point is elsewhere
        y_text = 10
        va = 'bottom'
    #####################################################
    
    plt.annotate(
        f'{model_display}{rerank_display}',
        (x, y),
        xytext=(x_text, y_text),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(
            facecolor='white',
            edgecolor=colors[entry['model']],
            alpha=1.0,
            pad=3,
            boxstyle='round,pad=0.5'
        ),
        horizontalalignment=ha,
        verticalalignment=va,
        clip_on=True,
        annotation_clip=True,
        zorder=3
    )

plt.xlabel('Estimated Cost ($)', fontsize=12, labelpad=10)
plt.ylabel('MRR@10', fontsize=12, labelpad=10)
plt.title(f'Performance vs Cost on Nano{args.dataset.upper()}', 
          fontsize=24, 
          fontweight='bold', 
          fontfamily='Crimson Pro',
          pad=20)
plt.gcf().patch.set_linewidth(2)
plt.gcf().patch.set_edgecolor('#0D4A34')
plt.tight_layout()
plt.savefig(f'performance_vs_cost_{args.dataset}.png', bbox_inches='tight', dpi=300, facecolor='#FEFBF6')
plt.close()
