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
    return f'latency_stats/{get_model_prefix(model)}_{dataset}_latency_stats.json'

def get_performance_file(model, dataset, rerank_key):
    rerank_part = "rerank_" if rerank_key == "rerank" else ""
    return f'performance_stats/{get_model_prefix(model)}_{dataset}_{rerank_part}performance_stats.json'

def get_model_prefix(model):
    prefixes = {
        'nomic-embed-text-v1.5': 'nomic-ai-nomic-embed-text-v1.5',
        'voyage-3-large': 'voyageai-voyage-3-large',
        'voyage-3': 'voyageai-voyage-3',
        'voyage-3-lite': 'voyageai-voyage-3-lite',
        'e5-mistral-7b-instruct': 'intfloat-e5-mistral-7b-instruct',
        'bge-m3': 'BAAI-bge-m3'
    }
    return prefixes[model]

model_names = [
    'nomic-embed-text-v1.5', 
    'voyage-3-large', 
    'voyage-3', 
    'voyage-3-lite', 
    # 'e5-mistral-7b-instruct',
    'bge-m3'
]
model_colors = {
    'nomic-embed-text-v1.5': '#179266', 
    'voyage-3-large': '#B80505',
    'voyage-3': '#E72727',
    'voyage-3-lite': '#FA8989',
    'e5-mistral-7b-instruct': '#FFC700',
    'bge-m3': '#FCBD88'
}

def load_data(datasets, metric, cost_per_million=False):
    """Load data for one or multiple datasets and return plot_data"""
    all_results = defaultdict(lambda: {
        'costs': [], 
        'metrics': [], 
        'total_cost': 0,
        'total_tokens': 0,
        'total_duration': 0
    })    
    if isinstance(datasets, str):
        datasets = [datasets]   
    for dataset in datasets:
        latency_stats = {}
        performance_stats = {}
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
        for model_name in model_names:
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
        for model, data in performance_stats.items():
            base_model_name = model.replace("-rerank", "")
            if base_model_name not in latency_stats:
                print(f"Warning: No latency data for {base_model_name}, skipping...")
                continue
            total_tokens = latency_stats[base_model_name]['total_tokens']
            processing_time = latency_stats[base_model_name]['total_duration_seconds']
            if 'voyage' in base_model_name.lower():
                if 'large' in base_model_name.lower():
                    embedding_cost = (total_tokens / 1_000_000) * VOYAGE_3_LARGE_COST_PER_MILLION_TOKENS
                    cost_per_million_tokens = VOYAGE_3_LARGE_COST_PER_MILLION_TOKENS
                elif 'lite' in base_model_name.lower():
                    embedding_cost = (total_tokens / 1_000_000) * VOYAGE_3_LITE_COST_PER_MILLION_TOKENS
                    cost_per_million_tokens = VOYAGE_3_LITE_COST_PER_MILLION_TOKENS
                else:
                    embedding_cost = (total_tokens / 1_000_000) * VOYAGE_3_COST_PER_MILLION_TOKENS
                    cost_per_million_tokens = VOYAGE_3_COST_PER_MILLION_TOKENS
            else:
                embedding_cost = processing_time * EC2_COST_PER_SECOND
                cost_per_million_tokens = (processing_time * EC2_COST_PER_SECOND) / (total_tokens / 1_000_000)
            total_cost = embedding_cost
            reranking_tokens = 0
            if data['use_reranker']:
                reranking_tokens = data.get('total_tokens_reranked', 0)
                reranking_cost = (reranking_tokens / 1_000_000) * VOYAGE_RERANK_COST_PER_MILLION_TOKENS
                total_cost += reranking_cost                
                if cost_per_million:
                    cost_per_million_tokens += VOYAGE_RERANK_COST_PER_MILLION_TOKENS
            cost_to_use = cost_per_million_tokens if cost_per_million else total_cost           
            key = f"{base_model_name}_{data['use_reranker']}"
            if len(datasets) == 1: 
                all_results[key] = {
                    'model': base_model_name,
                    metric: data[metric],
                    'cost': cost_to_use,
                    'use_reranker': data['use_reranker']
                }
            else:
                all_results[key]['costs'].append(cost_to_use if cost_per_million else 0)
                all_results[key]['metrics'].append(data[metric])
                all_results[key]['total_cost'] += total_cost
                all_results[key]['total_tokens'] += total_tokens
                all_results[key]['total_duration'] += processing_time
                if data['use_reranker']:
                    all_results[key]['total_reranking_tokens'] = all_results[key].get('total_reranking_tokens', 0) + reranking_tokens
                all_results[key]['model'] = base_model_name
                all_results[key]['use_reranker'] = data['use_reranker']
    plot_data = []
    if len(datasets) == 1:
        plot_data = list(all_results.values())
    else: 
        for key, stats in all_results.items():
            if len(stats['metrics']) > 0:
                if cost_per_million:
                    if 'voyage' in stats['model'].lower():
                        if 'large' in stats['model'].lower():
                            cost_per_million_val = VOYAGE_3_LARGE_COST_PER_MILLION_TOKENS
                        elif 'lite' in stats['model'].lower():
                            cost_per_million_val = VOYAGE_3_LITE_COST_PER_MILLION_TOKENS
                        else:
                            cost_per_million_val = VOYAGE_3_COST_PER_MILLION_TOKENS                        
                        if stats['use_reranker']:
                            cost_per_million_val += VOYAGE_RERANK_COST_PER_MILLION_TOKENS
                    else:
                        total_compute_cost = stats['total_duration'] * EC2_COST_PER_SECOND
                        cost_per_million_val = total_compute_cost / (stats['total_tokens'] / 1_000_000)
                        if stats['use_reranker'] and 'total_reranking_tokens' in stats:
                            cost_per_million_val += VOYAGE_RERANK_COST_PER_MILLION_TOKENS                   
                    avg_cost = cost_per_million_val
                else:
                    avg_cost = stats['total_cost']                    
                plot_data.append({
                    'model': stats['model'],
                    metric: sum(stats['metrics']) / len(stats['metrics']),
                    'cost': avg_cost,
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

    rerank_indicators = {False: 'o', True: '^'}
    model_points = {}
    for entry in plot_data:
        if entry['model'] not in model_points:
            model_points[entry['model']] = []
        model_points[entry['model']].append(entry)
    for model, points in model_points.items():
        if len(points) == 2:
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

    legend_handles = []
    for entry in plot_data:
        x = entry['cost']
        y = entry[metric]
        rerank_suffix = ' + rerank' if entry['use_reranker'] else ''
        label = f"{entry['model']}{rerank_suffix}"
        point = plt.scatter(
            x, y,
            c=model_colors[entry['model']],
            marker=rerank_indicators[entry['use_reranker']],
            s=50,
            label=label,
            zorder=2
        )
        legend_handles.append(point)

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
    plt.legend(
        handles=legend_handles,
        loc='best',
        fontsize=10,
        framealpha=0.9,
        edgecolor='#0D4A34',
        facecolor='white',
        bbox_to_anchor=(1.02, 1),
    )
    os.makedirs('plots', exist_ok=True)
    plt.gcf().patch.set_linewidth(2)
    plt.gcf().patch.set_edgecolor('#0D4A34')
    if x_axis_log:
        plt.xscale('log', base=10)
        plt.grid(True, which="both", ls="-", alpha=0.2)
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
    parser.add_argument('--cost_per_million', action='store_true', help='Plot cost per million tokens instead of total cost')

    args = parser.parse_args()
    cost_label = "Cost per Million Tokens ($)" if args.cost_per_million else "Estimated Cost ($)"
    avg_cost_label = "Cost per Million Tokens ($)" if args.cost_per_million else "Total Cost Across All Datasets ($)"
    if args.all:
        datasets = list(DATASET_PATHS.keys())
        for dataset in datasets:
            print(f"Creating plot for dataset: {dataset} with metric: {args.metric}")
            plot_data = load_data(dataset, args.metric, args.cost_per_million)
            title = f'Performance vs {cost_label.replace(" ($)", "")} on Nano{dataset.upper()}'
            filename = f'plots/{"cost_per_million" if args.cost_per_million else "performance_vs_cost"}_{dataset}_{args.metric}.png'
            create_plot(plot_data, title, args.metric, filename, args.log_x, cost_label=cost_label)
        print(f"Creating average plot for metric: {args.metric}")
        plot_data = load_data(list(DATASET_PATHS.keys()), args.metric, args.cost_per_million)
        title = f'Average NanoBEIR Performance vs {cost_label.replace(" ($)", "")}'
        filename = f'plots/average_{"cost_per_million" if args.cost_per_million else "performance_vs_cost"}_{args.metric}.png'
        create_plot(plot_data, title, args.metric, filename, args.log_x, cost_label=avg_cost_label)
    elif args.average:
        print(f"Creating average plot for metric: {args.metric}")
        plot_data = load_data(list(DATASET_PATHS.keys()), args.metric, args.cost_per_million)
        title = f'Average NanoBEIR Performance vs {cost_label.replace(" ($)", "")}'
        filename = f'plots/average_{"cost_per_million" if args.cost_per_million else "performance_vs_cost"}_{args.metric}.png'
        create_plot(plot_data, title, args.metric, filename, args.log_x, cost_label=avg_cost_label)
    else:
        print(f"Creating plot for dataset: {args.dataset} with metric: {args.metric}")
        plot_data = load_data(args.dataset, args.metric, args.cost_per_million)
        title = f'Performance vs {cost_label.replace(" ($)", "")} on Nano{args.dataset.upper()}'
        filename = f'plots/{"cost_per_million" if args.cost_per_million else "performance_vs_cost"}_{args.dataset}_{args.metric}.png'
        create_plot(plot_data, title, args.metric, filename, args.log_x, cost_label=cost_label)

if __name__ == "__main__":
    main()