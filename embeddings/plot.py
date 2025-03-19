import json
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
import argparse
import matplotlib.font_manager as fm
fm.fontManager.addfont("/Library/Fonts/Inter-VariableFont_opsz,wght.ttf")
fm.fontManager.addfont("/Library/Fonts/CrimsonPro-VariableFont_wght.ttf")

from corpus_embedding import DATASET_PATHS

parser = argparse.ArgumentParser(description='Plot performance vs cost for embedding models')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name or "all" for all datasets')
parser.add_argument('--metric', type=str, default='NDCG@10', choices=['NDCG@10', 'MRR@10'], help='Metric to plot')
args = parser.parse_args()

COLORS = ['#0D4A34', '#E5B300', '#0D4A34', '#179266', '#FA8989']
EC2_COST_PER_SECOND = 5.67 / 3600  # $5.67 per hour (G5.12xlarge cost) converted to per second

VOYAGE_RERANK_COST_PER_MILLION_TOKENS = 0.05
VOYAGE_3_LARGE_COST_PER_MILLION_TOKENS = 0.18
VOYAGE_3_COST_PER_MILLION_TOKENS = 0.06
VOYAGE_3_LITE_COST_PER_MILLION_TOKENS = 0.02

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

def format_cost(x, p):
    if x > 0:
        return f'${x:.2f}'
    else:
        return f'$0'

def create_plot(dataset, metric):
    print(f"Creating plot for dataset: {dataset} with metric: {metric}")
    latency_stats = {}
    performance_stats = {}
    plot_data = []

    model_to_latency_file = {
        # 'e5-mistral-7b-instruct': f'intfloat-e5-mistral-7b-instruct_{dataset}_latency_stats.json',
        'nomic-embed-text-v1.5': f'nomic-ai-nomic-embed-text-v1.5_{dataset}_latency_stats.json',
        'voyage-3-large': f'voyageai-voyage-3-large_{dataset}_latency_stats.json',
        'voyage-3': f'voyageai-voyage-3_{dataset}_latency_stats.json',
        'voyage-3-lite': f'voyageai-voyage-3-lite_{dataset}_latency_stats.json'
    }

    model_to_performance_file = {
        # 'e5-mistral-7b-instruct': {
        #     'default': f'intfloat-e5-mistral-7b-instruct_{dataset}_performance_stats.json',
        #     'rerank': f'intfloat-e5-mistral-7b-instruct_{dataset}_rerank_performance_stats.json'
        # },
        'nomic-embed-text-v1.5': {
            'default': f'nomic-ai-nomic-embed-text-v1.5_{dataset}_performance_stats.json',
            'rerank': f'nomic-ai-nomic-embed-text-v1.5_{dataset}_rerank_performance_stats.json'
        },
        'voyage-3-large': {
            'default': f'voyageai-voyage-3-large_{dataset}_performance_stats.json',
            'rerank': f'voyageai-voyage-3-large_{dataset}_rerank_performance_stats.json'
        },
        'voyage-3': {
            'default': f'voyageai-voyage-3_{dataset}_performance_stats.json',
            'rerank': f'voyageai-voyage-3_{dataset}_rerank_performance_stats.json'
        },
        'voyage-3-lite': {
            'default': f'voyageai-voyage-3-lite_{dataset}_performance_stats.json',
            'rerank': f'voyageai-voyage-3-lite_{dataset}_rerank_performance_stats.json'
        }
    }
    
    for model_name, latency_file in model_to_latency_file.items():
        file_path = os.path.join('latency_stats', latency_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                latency_stats[model_name] = {
                    'tokens_per_second': data['tokens_per_second'],
                    'total_tokens': data['total_tokens'],
                    'total_duration_seconds': data['total_duration_seconds']
                }

    for model_name, performance_dict in model_to_performance_file.items():
        # Load default (no reranker) stats
        default_file_path = os.path.join('performance_stats', performance_dict['default'])
        if os.path.exists(default_file_path):
            with open(default_file_path, 'r') as f:
                data = json.load(f)
                performance_stats[model_name] = {
                    'use_reranker': False,
                    'MRR@10': data['metrics']['MRR@10'],
                    'NDCG@10': data['metrics']['NDCG@10'],
                }
        else:
            print(f"Warning: Performance file not found: {default_file_path}")
            
        # Load reranker stats
        rerank_file_path = os.path.join('performance_stats', performance_dict['rerank'])
        if os.path.exists(rerank_file_path):
            with open(rerank_file_path, 'r') as f:
                data = json.load(f)
                performance_stats[model_name+"-rerank"] = {
                    'use_reranker': True,
                    'MRR@10': data['metrics']['MRR@10'],
                    'NDCG@10': data['metrics']['NDCG@10'],
                    'total_reranking_cost': data['metrics'].get('total_reranking_cost', 0),
                    'total_tokens_reranked': data['metrics'].get('total_tokens_reranked', 0),
                }
        else:
            print(f"Warning: Rerank performance file not found: {rerank_file_path}")

    # Calculate costs and prepare plot data
    for model, data in performance_stats.items():
        # Extract base model name by removing "-rerank" suffix if present
        base_model_name = model.replace("-rerank", "")
        
        # Skip if we don't have latency stats for this model
        if base_model_name not in latency_stats:
            print(f"Warning: No latency data for {base_model_name}, skipping...")
            continue
            
        if 'voyage' in base_model_name.lower():
            # For Voyage models, use API costs based on token usage
            total_tokens = latency_stats[base_model_name]['total_tokens']
            if 'large' in base_model_name.lower():
                embedding_cost = (total_tokens / 1_000_000) * VOYAGE_3_LARGE_COST_PER_MILLION_TOKENS
            elif 'lite' in base_model_name.lower():
                embedding_cost = (total_tokens / 1_000_000) * VOYAGE_3_LITE_COST_PER_MILLION_TOKENS
            else:
                embedding_cost = (total_tokens / 1_000_000) * VOYAGE_3_COST_PER_MILLION_TOKENS
        else:
            # For Nomic and e5-mistral models, use compute costs
            processing_time = latency_stats[base_model_name]['total_duration_seconds']
            embedding_cost = processing_time * EC2_COST_PER_SECOND
        
        total_cost = embedding_cost
        if data['use_reranker']:
            # Get reranking cost from the rerank variant's data
            reranking_tokens = data.get('total_tokens_reranked', 0)
            reranking_cost = (reranking_tokens / 1_000_000) * VOYAGE_RERANK_COST_PER_MILLION_TOKENS
            total_cost += reranking_cost
            
        plot_data.append({
            'model': base_model_name,
            metric: data[metric],
            'cost': total_cost,
            'use_reranker': data['use_reranker']
        })

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(format_cost))
    ax.grid(True, which='major', linestyle='--', alpha=0.2)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1)
    ax.minorticks_on()

    colors = {
        'nomic-embed-text-v1.5': '#179266', 
        'voyage-3-large': '#CC9F00',
        'voyage-3': '#E5B300',
        'voyage-3-lite': '#FFC700', 
        # 'e5-mistral-7b-instruct': '#FA8989'
    }
    markers = {False: 'o', True: '^'}
    model_points = {}
    
    # Group by model
    for entry in plot_data:
        if entry['model'] not in model_points:
            model_points[entry['model']] = []
        model_points[entry['model']].append(entry)
    
    # Draw connecting lines between regular and reranked versions
    for model, points in model_points.items():
        if len(points) == 2:  # Only if we have both regular and reranked
            points.sort(key=lambda x: x['use_reranker'])
            plt.plot(
                [p['cost'] for p in points],
                [p[metric] for p in points],
                color=colors[model],
                alpha=0.3,
                linestyle='--',
                zorder=1
            )

    # Plot points
    for entry in plot_data:
        x = entry['cost']
        y = entry[metric]
        plt.scatter(
            x, y,
            c=colors[entry['model']],
            marker=markers[entry['use_reranker']],
            s=150,
            label=f"{entry['model']} (rerank: {entry['use_reranker']})",
            zorder=2
        )
        
        rerank_display = '\n+ voyageai/rerank-2' if entry['use_reranker'] else ''
        
        ###################################################################
        # Make sure annotations don't go outside the plot
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
        ###################################################################
        
        plt.annotate(
            f'{entry["model"]}{rerank_display}',
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

    # Set left x-limit to 0 while preserving right limit
    _, x_max = plt.xlim()
    plt.xlim(0, x_max)

    plt.xlabel('Estimated Cost ($)', fontsize=12, labelpad=10)
    plt.ylabel(metric, fontsize=12, labelpad=10)  # Use the actual metric name for the y-axis
    plt.title(f'Performance vs Cost on Nano{dataset.upper()}', 
              fontsize=24, 
              fontweight='bold', 
              fontfamily='Crimson Pro',
              pad=20)
    
    # Ensure directory exists
    os.makedirs('plots', exist_ok=True)
    
    plt.gcf().patch.set_linewidth(2)
    plt.gcf().patch.set_edgecolor('#0D4A34')
    plt.tight_layout()
    plt.savefig(f'plots/performance_vs_cost_{dataset}_{metric}.png', bbox_inches='tight', dpi=300, facecolor='#FEFBF6')
    plt.close()

def main():
    if args.dataset == 'all':
        datasets = list(DATASET_PATHS.keys())
        for dataset in datasets:
            create_plot(dataset, args.metric)
    else:
        create_plot(args.dataset, args.metric)

if __name__ == "__main__":
    main()
