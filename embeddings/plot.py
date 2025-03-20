import json
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
import argparse
import matplotlib.font_manager as fm
from collections import defaultdict
fm.fontManager.addfont("/Library/Fonts/Inter-VariableFont_opsz,wght.ttf")
fm.fontManager.addfont("/Library/Fonts/CrimsonPro-VariableFont_wght.ttf")

from corpus_embedding import DATASET_PATHS

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
    
def get_latency_file(model, dataset):
    model_to_latency_file = {
        'nomic-embed-text-v1.5': f'nomic-ai-nomic-embed-text-v1.5_{dataset}_latency_stats.json',
        'voyage-3-large': f'voyageai-voyage-3-large_{dataset}_latency_stats.json',
        'voyage-3': f'voyageai-voyage-3_{dataset}_latency_stats.json',
        'voyage-3-lite': f'voyageai-voyage-3-lite_{dataset}_latency_stats.json'
    }
    return'latency_stats/'+model_to_latency_file[model]

def get_performance_file(model, dataset, rerank_key):
    model_to_performance_file = {
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
    return 'performance_stats/'+model_to_performance_file[model][rerank_key]

model_names = ['nomic-embed-text-v1.5', 'voyage-3-large', 'voyage-3', 'voyage-3-lite']
model_colors = {
    'nomic-embed-text-v1.5': '#179266', 
    'voyage-3-large': '#B80505',
    'voyage-3': '#E72727',
    'voyage-3-lite': '#FA8989',
    'e5-mistral-7b-instruct': '#FFC700'
}

def load_data(datasets, metric):
    """Load data for one or multiple datasets and return plot_data"""
    all_results = defaultdict(lambda: {'costs': [], 'metrics': [], 'total_cost': 0})
    
    # Handle single dataset case
    if isinstance(datasets, str):
        datasets = [datasets]
        
    for dataset in datasets:
        latency_stats = {}
        performance_stats = {}
        
        # Load latency stats
        for model_name in model_names:
            file_path = get_latency_file(model_name, dataset)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    latency_stats[model_name] = {
                        'tokens_per_second': data['tokens_per_second'],
                        'total_tokens': data['total_tokens'],
                        'total_duration_seconds': data['total_duration_seconds']
                    }

        # Load performance stats (default and rerank)
        for model_name in model_names:
            # Default (no reranker) stats
            default_file_path = get_performance_file(model_name, dataset, 'default')
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
                
            # Reranker stats
            rerank_file_path = get_performance_file(model_name, dataset, 'rerank')
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

        # Calculate costs for each model/variant
        for model, data in performance_stats.items():
            base_model_name = model.replace("-rerank", "")
            if base_model_name not in latency_stats:
                print(f"Warning: No latency data for {base_model_name}, skipping...")
                continue
                
            # Calculate embedding cost based on model type
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
                # For Nomic models, use compute costs
                processing_time = latency_stats[base_model_name]['total_duration_seconds']
                embedding_cost = processing_time * EC2_COST_PER_SECOND
            
            # Add reranking cost if applicable
            total_cost = embedding_cost
            if data['use_reranker']:
                reranking_tokens = data.get('total_tokens_reranked', 0)
                reranking_cost = (reranking_tokens / 1_000_000) * VOYAGE_RERANK_COST_PER_MILLION_TOKENS
                total_cost += reranking_cost
            
            # Store results for single dataset or average calculation
            if len(datasets) == 1:  # Single dataset
                key = f"{base_model_name}_{data['use_reranker']}"
                all_results[key] = {
                    'model': base_model_name,
                    metric: data[metric],
                    'cost': total_cost,
                    'use_reranker': data['use_reranker']
                }
            else:  # Multiple datasets for averaging
                key = f"{base_model_name}_{data['use_reranker']}"
                all_results[key]['costs'].append(total_cost)
                all_results[key]['metrics'].append(data[metric])
                all_results[key]['total_cost'] += total_cost
                all_results[key]['model'] = base_model_name
                all_results[key]['use_reranker'] = data['use_reranker']

    # Convert results to plot data format
    plot_data = []
    if len(datasets) == 1:  # Single dataset mode
        plot_data = list(all_results.values())
    else:  # Average mode
        for key, stats in all_results.items():
            if len(stats['metrics']) > 0:
                plot_data.append({
                    'model': stats['model'],
                    metric: sum(stats['metrics']) / len(stats['metrics']),  # Average metric
                    'cost': stats['total_cost'],  # Total cost across all datasets
                    'use_reranker': stats['use_reranker']
                })

    return plot_data

def create_plot(plot_data, title, metric, filename, x_axis_log, cost_label="Estimated Cost ($)"):
    """Create and save plot with given data"""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(format_cost))
    ax.grid(True, which='major', linestyle='--', alpha=0.2)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1)
    ax.minorticks_on()

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
                color=model_colors[model],
                alpha=0.3,
                linestyle='--',
                zorder=1
            )
        elif len(points) == 1:
            print(f"Only one variant found for {model}")
        else:
            print(f"Warning: Unexpected number of points ({len(points)}) for {model}")

    # Plot points with proper legend entries
    legend_handles = []
    
    for entry in plot_data:
        x = entry['cost']
        y = entry[metric]
        
        # Create label for legend
        rerank_suffix = ' + rerank' if entry['use_reranker'] else ''
        label = f"{entry['model']}{rerank_suffix}"
        
        # Plot point
        point = plt.scatter(
            x, y,
            c=model_colors[entry['model']],
            marker=markers[entry['use_reranker']],
            s=150,
            label=label,
            zorder=2
        )
        legend_handles.append(point)

    # Set left x-limit to 0 while preserving right limit
    if not x_axis_log:
        _, x_max = plt.xlim()
        plt.xlim(0, x_max)

        plt.xlabel(cost_label, fontsize=12, labelpad=10)
    else:
        plt.xlabel("log cost", fontsize=12, labelpad=10)
    plt.ylabel(metric, fontsize=12, labelpad=10)
    plt.title(title, 
          fontsize=24, 
          fontweight='bold', 
          fontfamily='Crimson Pro',
          pad=20)
    
    # Add legend with good formatting
    plt.legend(
        handles=legend_handles,
        loc='best',
        fontsize=10,
        framealpha=0.9,
        edgecolor='#0D4A34',
        facecolor='white',
        bbox_to_anchor=(1.02, 1),
    )
    
    # Ensure directory exists
    os.makedirs('plots', exist_ok=True)
    
    plt.gcf().patch.set_linewidth(2)
    plt.gcf().patch.set_edgecolor('#0D4A34')

    if x_axis_log:
        plt.xscale('log', base=10)
        # Add grid lines for log scale
        plt.grid(True, which="both", ls="-", alpha=0.2)
        # Format x-axis tick labels to be more readable
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
        filename = filename.replace("plots/", "plots/log_")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='#FEFBF6')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot performance vs cost for embedding models')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--metric', type=str, default='NDCG@10', choices=['NDCG@10', 'MRR@10'], help='Metric to plot')
    parser.add_argument('--average', action='store_true', help='Plot average metrics across all datasets')
    parser.add_argument('--all', action='store_true', help='Plot all datasets')
    parser.add_argument('--log_x', action='store_true', help='Make X axis logarithmic')

    args = parser.parse_args()
    if args.all:
        datasets = list(DATASET_PATHS.keys())
        for dataset in datasets:
            print(f"Creating plot for dataset: {dataset} with metric: {args.metric}")
            plot_data = load_data(dataset, args.metric)
            title = f'Performance vs Cost on Nano{dataset.upper()}'
            filename = f'plots/performance_vs_cost_{dataset}_{args.metric}.png'
            create_plot(plot_data, title, args.metric, filename, args.log_x)
        print(f"Creating average plot for metric: {args.metric}")
        plot_data = load_data(list(DATASET_PATHS.keys()), args.metric)
        title = f'Average NanoBEIR Performance vs Cost'
        filename = f'plots/average_performance_vs_cost_{args.metric}.png'
        create_plot(plot_data, title, args.metric, filename, args.log_x, cost_label="Total Cost Across All Datasets ($)")
    elif args.average:
        print(f"Creating average plot for metric: {args.metric}")
        plot_data = load_data(list(DATASET_PATHS.keys()), args.metric)
        title = f'Average NanoBEIR Performance vs Cost'
        filename = f'plots/average_performance_vs_cost_{args.metric}.png'
        create_plot(plot_data, title, args.metric, filename, args.log_x, cost_label="Total Cost Across All Datasets ($)")
    else:
        print(f"Creating plot for dataset: {args.dataset} with metric: {args.metric}")
        plot_data = load_data(args.dataset)
        title = f'Performance vs Cost on Nano{args.dataset.upper()}'
        filename = f'plots/performance_vs_cost_{args.dataset}_{args.metric}.png'
        create_plot(plot_data, title, args.metric, filename, args.log_x)

if __name__ == "__main__":
    main()