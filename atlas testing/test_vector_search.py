import os
import requests
from nomic import embed
import argparse
import time

API_KEY = os.environ.get('NOMIC_API_KEY')
DEFAULT_DATASET = "nomic/federal-register"
DEFAULT_PROJECTION_ID = "89317fbc-3496-42d4-8cd0-0fe35a4dda8f"
DEFAULT_INDEX_ID = "b84d4b30-3752-4271-a703-98cb1dd75b9d"
DEFAULT_QUERY = "raw food disease"
DEFAULT_K = 100

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

def get_embedding(query):
    emb_vec = embed.text([query])['embeddings']
    print("embedding shape:", len(emb_vec))
    return emb_vec

def get_topk_results(embedding, projection_id, k_val):
    url = "https://api-atlas.nomic.ai/v1/query/topk"
    payload = {
        "projection_id": projection_id,
        "k": k_val,
        "query": embedding[0]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def get_nearest_neighbors(embedding, index_id, k_val):
    url = "https://api-atlas.nomic.ai/v1/project/data/get/nearest_neighbors/by_embedding"
    payload = {
        "atlas_index_id": index_id,
        "queries": embedding,
        "k": k_val
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def _process_topk_item_comparison(
    topk_item, current_index_i, rank, nn_rank_map, nn_dist_map,
    swapped_pair_first_indices, topk_data, nn_neighbors, k_val_from_args, eff_k_val
):
    """Utility function to compute how different two rankings are from ranking endpoints"""
    current_topk_id = str(topk_item['id'])
    topk_score = topk_item['_similarity']

    status = ""
    nn_score_display = "---"
    displacement_display = "---"
    summary_details = None
    ordered_match_delta = 0
    is_swap_flag = False # Local flag

    if current_topk_id in nn_rank_map:
        actual_nn_rank = nn_rank_map[current_topk_id]
        nn_score = nn_dist_map[current_topk_id]
        nn_score_display = f"{nn_score:.4f}"
        displacement_val = actual_nn_rank - rank
        displacement_display = f"{displacement_val:+}" if displacement_val != 0 else ""

        if displacement_val == 0:
            status = "✅"
            if rank <= eff_k_val:
                ordered_match_delta = 1
        else: 
            # Check for current item starting a swap
            if displacement_val == 1 and (current_index_i + 1) < k_val_from_args:
                next_topk_item_idx = current_index_i + 1
                if next_topk_item_idx < len(topk_data):
                    next_topk_item = topk_data[next_topk_item_idx]
                    if next_topk_item and 'id' in next_topk_item:
                        next_topk_id = str(next_topk_item['id'])
                        if next_topk_id in nn_rank_map:
                            next_topk_actual_nn_rank = nn_rank_map[next_topk_id]
                            if (next_topk_actual_nn_rank - (rank + 1)) == -1:
                                nn_id_at_current_rank_for_swap = str(nn_neighbors[current_index_i]) if current_index_i < len(nn_neighbors) else None
                                nn_id_at_next_rank_for_swap = str(nn_neighbors[current_index_i+1]) if (current_index_i+1) < len(nn_neighbors) else None
                                if current_topk_id == nn_id_at_next_rank_for_swap and next_topk_id == nn_id_at_current_rank_for_swap:
                                    status = "Swap"
                                    swapped_pair_first_indices.add(current_index_i)
                                    is_swap_flag = True
            # Check if current item is completing a previously initiated swap
            elif displacement_val == -1 and current_index_i > 0 and (current_index_i - 1) in swapped_pair_first_indices:
                status = "Swap"
                is_swap_flag = True 
            
            if not is_swap_flag: # If not part of any swap, it's a Mismatch
                status = "Mismatch"
            
            summary_details = {
                'id': current_topk_id, 'topk_rank': rank, 'topk_score': topk_score,
                'nn_rank': actual_nn_rank, 'nn_score': nn_score,
                'is_swap': is_swap_flag 
            }
    else: 
        status = "TopK Only"
        summary_details = {
            'id': current_topk_id, 'topk_rank': rank, 'topk_score': topk_score,
            'is_topk_only': True
        }
    
    return status, nn_score_display, displacement_display, summary_details, ordered_match_delta

def main(args):
    print(f"Testing vector search on {args.dataset} with query: '{args.query}' for K={args.k}")
    print("\nGetting embedding for the query...")
    embedding_start_time = time.time()
    embedding = get_embedding(args.query)
    embedding_latency = time.time() - embedding_start_time
    print(f"Embedding generation took: {embedding_latency:.4f}s")

    print("\nFetching results from query/topk endpoint...")
    topk_start_time = time.time()
    topk_results = get_topk_results(embedding, args.projection_id, args.k)
    topk_latency = time.time() - topk_start_time
    print(f"query/topk endpoint took: {topk_latency:.4f}s")

    print("\nFetching results from nearest_neighbors/by_embedding endpoint...")
    nn_start_time = time.time()
    nn_results = get_nearest_neighbors(embedding, args.index_id, args.k)
    nn_latency = time.time() - nn_start_time
    print(f"nearest_neighbors/by_embedding endpoint took: {nn_latency:.4f}s")

    print("\nComparing the results up to K=", args.k)
    print("="*90)
    topk_data = topk_results.get("data", [])
    nn_neighbors = nn_results.get("neighbors", [[]])[0]
    nn_distances = nn_results.get("distances", [[]])[0]

    nn_rank_map = {str(nn_id): i + 1 for i, nn_id in enumerate(nn_neighbors)}
    nn_dist_map = {str(nn_id): nn_distances[i] for i, nn_id in enumerate(nn_neighbors)}
    topk_rank_map = {str(topk_data[i]['id']): i + 1 for i in range(len(topk_data)) if topk_data[i] and 'id' in topk_data[i]}

    mismatches_for_summary = []
    ordered_matches = 0
    eff_k = min(args.k, len(topk_data), len(nn_neighbors))
    
    swapped_pair_first_indices = set()
    
    print("\nThe 'Rank' and 'ID' columns reflect the order from the 'query/topk' endpoint.\n")
    print(f"{'Rank':<5} | {'ID':<20} | {'Score (TopK)':<15} | {'Score (NN)':<15} | {'NN Rank Δ':<10} | {'Result':<15}")
    print("-"*90)

    for i in range(args.k):
        rank = i + 1
        topk_item = topk_data[i] if i < len(topk_data) else None
        nn_item_at_this_rank_id = str(nn_neighbors[i]) if i < len(nn_neighbors) else None

        display_id = "---"
        display_topk_score = "---"
        display_nn_score = "---"
        display_displacement = "---"
        display_status = ""

        if topk_item and 'id' in topk_item:
            display_id = str(topk_item['id'])
            display_topk_score = f"{topk_item['_similarity']:.4f}"

            status, nn_s, disp_s, summary, ord_match_inc = _process_topk_item_comparison(
                topk_item, i, rank, nn_rank_map, nn_dist_map,
                swapped_pair_first_indices, topk_data, nn_neighbors, args.k, eff_k
            )
            display_status = status
            display_nn_score = nn_s
            display_displacement = disp_s
            if summary:
                mismatches_for_summary.append(summary)
            ordered_matches += ord_match_inc
        
        elif nn_item_at_this_rank_id:
            display_id = f"(NN) {nn_item_at_this_rank_id}"
            display_nn_score = f"{nn_distances[i]:.4f}" if i < len(nn_distances) else "---"
            display_status = "NN Only@Rank"
        
        else: 
            if rank <= args.k:
                display_id = "N/A"
                display_status = "Both Short"
            else:
                continue 

        print(f"{rank:<5} | {display_id:<20} | {display_topk_score:<15} | {display_nn_score:<15} | {display_displacement:<10} | {display_status:<15}")

    topk_ids_set_for_summary = {str(item['id']) for item in topk_data[:args.k] if item and 'id' in item}
    nn_only_ids_in_k_for_summary = []
    for i_nn in range(min(args.k, len(nn_neighbors))):
        nn_id = str(nn_neighbors[i_nn])
        if nn_id not in topk_ids_set_for_summary:
            nn_only_ids_in_k_for_summary.append({'id': nn_id, 'nn_rank': i_nn + 1, 'nn_score': nn_distances[i_nn]})

    print("-"*90)
    print("\nSummary of Comparison (up to K=", args.k, "):")
    print(f"- Matched in order (Disp. 0): {ordered_matches} / {min(args.k, len(topk_data), len(nn_neighbors)) if min(len(topk_data), len(nn_neighbors)) > 0 else args.k}")
    
    num_swapped_pairs = len(swapped_pair_first_indices)
    print(f"- Adjacent Swaps: {num_swapped_pairs} pairs")
    
    true_mismatches_summary = [
        m for m in mismatches_for_summary 
        if not m.get('is_swap') and not m.get('is_topk_only') and 'nn_rank' in m
    ]
    
    print("="*90)
    print(f"API Latencies:")
    print(f"  - Embedding generation: {embedding_latency:.4f}s")
    print(f"  - query/topk: {topk_latency:.4f}s")
    print(f"  - nearest_neighbors/by_embedding: {nn_latency:.4f}s")
    print("="*90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Nomic vector search endpoints.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"Dataset name (default: {DEFAULT_DATASET})")
    parser.add_argument("--projection_id", type=str, default=DEFAULT_PROJECTION_ID, help=f"Projection ID for topk endpoint (default: {DEFAULT_PROJECTION_ID})")
    parser.add_argument("--index_id", type=str, default=DEFAULT_INDEX_ID, help=f"Atlas Index ID for nearest_neighbors endpoint (default: {DEFAULT_INDEX_ID})")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help=f"Query string (default: '{DEFAULT_QUERY}')")
    parser.add_argument("-k", "--k", type=int, default=DEFAULT_K, help=f"Number of results to fetch (default: {DEFAULT_K})")
    
    parsed_args = parser.parse_args()
    main(parsed_args)
