import argparse
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import logging
from transformers import AutoTokenizer
from query_embedding_and_metrics import DATASET_PATHS, EmbeddingEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_retrieved_documents(model1_name, model2_name, dataset_name, initial_k=100, num_queries=50):
    """Analyze differences in retrieved documents between two models."""
    
    # Initialize evaluators for both models
    evaluator1 = EmbeddingEvaluator(model1_name, dataset_name, k=10, use_reranker=False, initial_k=initial_k)
    evaluator2 = EmbeddingEvaluator(model2_name, dataset_name, k=10, use_reranker=False, initial_k=initial_k)
    
    # Use the same rerank tokenizer for consistent token counting
    tokenizer = AutoTokenizer.from_pretrained("voyageai/rerank-2")
    
    # Get a subset of queries to analyze
    query_ids = list(evaluator1.queries.keys())[:num_queries]
    
    stats = {
        "model1": {"name": model1_name, "total_tokens": 0, "doc_lengths": []},
        "model2": {"name": model2_name, "total_tokens": 0, "doc_lengths": []},
        "overlap": [],
    }
    
    for query_id in tqdm(query_ids, desc="Analyzing queries"):
        query_text = evaluator1.queries[query_id]
        query_tokens = len(tokenizer.encode(query_text, add_special_tokens=True))
        
        # Get embeddings and retrieve documents for both models
        for evaluator, model_key in [(evaluator1, "model1"), (evaluator2, "model2")]:
            query_embedding = evaluator.embed_queries([query_text], [query_id])[query_id]
            
            # Calculate similarities and get top documents
            similarities = []
            for doc_id, doc_embedding in evaluator.corpus_embeddings.items():
                query_norm = np.linalg.norm(query_embedding)
                query_embedding_norm = query_embedding / query_norm if query_norm > 0 else query_embedding
                
                doc_norm = np.linalg.norm(doc_embedding)
                doc_embedding_norm = doc_embedding / doc_norm if doc_norm > 0 else doc_embedding
                
                similarity = np.dot(query_embedding_norm, doc_embedding_norm)
                similarities.append((similarity, doc_id))
            
            similarities.sort(reverse=True)
            top_docs = [(score, doc_id) for score, doc_id in similarities[:initial_k]]
            
            # Calculate token counts for retrieved documents
            docs_to_analyze = [evaluator.corpus[doc_id] for _, doc_id in top_docs]
            doc_token_counts = [len(tokenizer.encode(doc, add_special_tokens=True)) for doc in docs_to_analyze]
            
            # Update stats
            stats[model_key]["doc_lengths"].extend(doc_token_counts)
            total_tokens = sum(doc_token_counts) + (query_tokens * len(docs_to_analyze))
            stats[model_key]["total_tokens"] += total_tokens
            
            # Store document IDs for overlap analysis
            if model_key == "model1":
                model1_docs = set(doc_id for _, doc_id in top_docs)
            else:
                model2_docs = set(doc_id for _, doc_id in top_docs)
                overlap = len(model1_docs.intersection(model2_docs))
                stats["overlap"].append(overlap / initial_k)
    
    # Calculate aggregate statistics and convert numpy types to native Python types
    for model_key in ["model1", "model2"]:
        doc_lengths = stats[model_key]["doc_lengths"]
        stats[model_key].update({
            "avg_doc_length": float(np.mean(doc_lengths)),
            "median_doc_length": float(np.median(doc_lengths)),
            "min_doc_length": int(np.min(doc_lengths)),
            "max_doc_length": int(np.max(doc_lengths)),
            "std_doc_length": float(np.std(doc_lengths)),
            "total_tokens": int(stats[model_key]["total_tokens"]),
        })
    
    # Calculate average overlap
    stats["average_overlap"] = float(np.mean(stats["overlap"]))
    stats["overlap"] = [float(x) for x in stats["overlap"]]
    

    del stats["model1"]["doc_lengths"]
    del stats["model2"]["doc_lengths"]
    del stats["overlap"]
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Analyze tokenization differences between models")
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="fever")
    parser.add_argument("--initial_k", type=int, default=100)
    parser.add_argument("--num_queries", type=int, default=50)
    parser.add_argument("--output", type=str, default="tokenization_analysis.json")
    args = parser.parse_args()
    
    logger.info(f"Analyzing tokenization differences between {args.model1} and {args.model2}")
    stats = analyze_retrieved_documents(
        args.model1,
        args.model2,
        args.dataset,
        initial_k=args.initial_k,
        num_queries=args.num_queries
    )
    
    # Print summary statistics
    for model_key in ["model1", "model2"]:
        model_name = stats[model_key]["name"]
        logger.info(f"\nStatistics for {model_name}:")
        logger.info(f"Total tokens: {stats[model_key]['total_tokens']:,}")
        logger.info(f"Average document length: {stats[model_key]['avg_doc_length']:.2f} tokens")
        logger.info(f"Median document length: {stats[model_key]['median_doc_length']:.2f} tokens")
        logger.info(f"Min document length: {stats[model_key]['min_doc_length']} tokens")
        logger.info(f"Max document length: {stats[model_key]['max_doc_length']} tokens")
        logger.info(f"Standard deviation: {stats[model_key]['std_doc_length']:.2f} tokens")
    
    logger.info(f"\nAverage document overlap between models: {stats['average_overlap']*100:.2f}%")
    
    # Save detailed statistics
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nDetailed statistics saved to {args.output}")

if __name__ == "__main__":
    main() 