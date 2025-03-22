import argparse
import glob
import json
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import requests
import voyageai


from corpus_embedding import DATASET_PATHS

VOYAGE_RERANK_COST_PER_MILLION_TOKENS = 0.05

e5_mistral_prompt_dict = {
    "arguana": "Given a claim, find documents that refute the claim: ",
    "climate-fever": "Given a claim about climate change, retrieve documents that support or refute the claim: ",
    "dbpedia": "Given a query, retrieve relevant entity descriptions from DBPedia: ",
    "fever": "Given a claim, retrieve documents that support or refute the claim: ",
    "fiqa": "Given a financial question, retrieve user replies that best answer the question: ",
    "hotpot": "Given a multi-hop question, retrieve documents that can help answer the question: ",
    "msmarco": "Given a web search query, retrieve relevant passages that answer the query: ",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question: ",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question: ",
    "quora": "Given a question, retrieve questions that are semantically equivalent to the given question: ",
    "scidocs": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper: ",
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim: ",
    "touche": "Given a question, retrieve detailed and persuasive arguments that answer the question: ",
}

def compute_dcg_at_k(relevances, k):
    dcg = 0
    for i in range(min(len(relevances), k)):
        dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
    return dcg

class EmbeddingEvaluator:
    def __init__(self, model_name, dataset_name, endpoint="", k=10, use_reranker=False, initial_k=100):
        """Initialize the embedding evaluator"""
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json"}
        self.k = k
        self.use_reranker = use_reranker
        self.initial_k = initial_k if use_reranker else k
        if "nomic" in model_name:
            self.query_prefix="search_query: "
        elif "e5-mistral" in model_name:
            self.query_prefix = e5_mistral_prompt_dict[dataset_name]
        else:
            self.query_prefix = ""
        
        self.total_reranking_cost = 0.0
        self.total_tokens_reranked = 0
        
        self.use_voyage = "voyage" in model_name.lower()
        voyage_api_key = os.environ.get("VOYAGE_API_KEY", None)
        if not voyage_api_key and (use_reranker or self.use_voyage):
            raise ValueError("Voyage API key required for reranking or Voyage models")
        if use_reranker or self.use_voyage:
            voyageai.api_key = voyage_api_key
            self.vo = voyageai.Client()
        
        self.corpus, self.queries, self.qrels = self._load_dataset()
        self.corpus_embeddings = self._load_cached_embeddings()
        
    def _load_dataset(self):
        """Load and process the dataset splits"""
        if self.dataset_name not in DATASET_PATHS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. Available: {list(DATASET_PATHS.keys())}")
        
        try:
            dataset_path = DATASET_PATHS[self.dataset_name]
            corpus = load_dataset(dataset_path, "corpus", split="train")
            queries = load_dataset(dataset_path, "queries", split="train")
            qrels = load_dataset(dataset_path, "qrels", split="train")
            processed_corpus = {str(s["_id"]): s["text"] 
                              for s in corpus if len(s["text"].strip()) > 0}
            processed_queries = {str(s["_id"]): s["text"] 
                               for s in queries if len(s["text"].strip()) > 0}
            
            processed_qrels = {}
            for item in qrels:
                query_id = str(item["query-id"])
                corpus_id = str(item["corpus-id"])
                if query_id not in processed_qrels:
                    processed_qrels[query_id] = set()
                processed_qrels[query_id].add(corpus_id)
            
            missing_docs = set()
            for query_id, relevant_docs in processed_qrels.items():
                for doc_id in list(relevant_docs):
                    if doc_id not in processed_corpus:
                        relevant_docs.remove(doc_id)
                        missing_docs.add(doc_id)
            
            if missing_docs:
                logger.warning(f"Found {len(missing_docs)} referenced documents that don't exist in corpus")
                processed_qrels = {qid: docs for qid, docs in processed_qrels.items() if docs}
            
            logger.info(f"Dataset loaded: {len(processed_corpus)} docs, {len(processed_queries)} queries")
            return processed_corpus, processed_queries, processed_qrels
            
        except Exception as e:
            raise ValueError(f"Error loading dataset {self.dataset_name}: {str(e)}")
    
    def _load_cached_embeddings(self):
        """Load cached embeddings from the cache directory"""
        cache_dir = "embeddings_cache"
        model_name_safe = self.model_name.replace("/", "-")

        cache_file = f"{cache_dir}/{model_name_safe}_{self.dataset_name}_embeddings"
        
        if not os.path.exists(cache_file + '.npy') or not os.path.exists(cache_file + '_ids.json'):
            raise FileNotFoundError(f"Cached embeddings not found at {cache_file}")
        embeddings_array = np.load(cache_file + '.npy')
        with open(cache_file + '_ids.json', 'r') as f:
            doc_ids = json.load(f)
        embeddings_dict = {doc_id: embeddings_array[i] for i, doc_id in enumerate(doc_ids)}
        logger.info(f"Loaded {len(embeddings_dict)} cached embeddings")
        return embeddings_dict

    def _load_cached_query_embeddings(self):
        """Load cached query embeddings if they exist"""
        cache_dir = "embeddings_cache"
        model_name_safe = self.model_name.replace("/", "-")
        cache_file = f"{cache_dir}/{model_name_safe}_{self.dataset_name}_queries"
        
        if os.path.exists(cache_file + '.npy') and os.path.exists(cache_file + '_ids.json'):
            embeddings_array = np.load(cache_file + '.npy')
            with open(cache_file + '_ids.json', 'r') as f:
                query_ids = json.load(f)
            
            embeddings_dict = {qid: embeddings_array[i] for i, qid in enumerate(query_ids)}
            logger.info(f"Loaded {len(embeddings_dict)} cached query embeddings")
            return embeddings_dict
        return None

    def _save_query_embeddings(self, query_embeddings):
        """Save query embeddings to cache"""
        cache_dir = "embeddings_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        model_name_safe = self.model_name.replace("/", "-")
        cache_file = f"{cache_dir}/{model_name_safe}_{self.dataset_name}_queries"
        
        query_ids = list(query_embeddings.keys())
        embeddings_array = np.array([query_embeddings[qid] for qid in query_ids])
        
        np.save(cache_file + '.npy', embeddings_array)
        with open(cache_file + '_ids.json', 'w') as f:
            json.dump(query_ids, f)
        logger.info(f"Saved {len(query_embeddings)} query embeddings to cache")
    
    def embed_queries(self, query_texts, query_ids):
        """Embed multiple queries at once"""
        cached_embeddings = self._load_cached_query_embeddings()
        if cached_embeddings is not None:
            return cached_embeddings
        query_embeddings = {}
        try:
            if self.use_voyage:
                result = self.vo.embed(
                    query_texts, 
                    model=self.model_name.replace("voyageai/", ""), 
                    input_type="query"
                )
                embeddings = result.embeddings
            else:
                if "e5-mistral" in self.model_name:
                    payload = {
                        "inputs": [
                            e5_mistral_prompt_dict[self.dataset_name] + x
                            for x in query_texts
                        ]
                    }
                else:
                    payload = {"inputs": [self.query_prefix + x for x in query_texts]}
                response = requests.post( 
                    f"{self.endpoint}/embed",
                    headers=self.headers,
                    json=payload
                )
                if response.status_code != 200:
                    raise Exception(f"Embedding request failed: {response.status_code}")
                embeddings = response.json()
            
            query_embeddings = {qid: emb for qid, emb in zip(query_ids, embeddings)}
            self._save_query_embeddings(query_embeddings)
            return query_embeddings
        except Exception as e:
            logger.error(f"Error embedding queries: {str(e)}")
            return None
    
    def _get_metrics_cache_path(self, query_id):
        """Get cache path for metrics results"""
        cache_dir = "metrics_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        model_name_safe = self.model_name.replace("/", "-")
        rerank_suffix = f"_rerank{self.k}_from{self.initial_k}" if self.use_reranker else f"_k{self.k}"
        return f"{cache_dir}/{model_name_safe}_{self.dataset_name}_{query_id}{rerank_suffix}.json"

    def calculate_metrics(self, query_embedding, query_id):
        """Calculate metrics for a single query with caching"""
        cache_path = self._get_metrics_cache_path(query_id)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_results = json.load(f)
                    logger.debug(f"Loaded cached metrics for query {query_id}")
                    if self.use_reranker and "tokens_reranked" in cached_results:
                        self.total_tokens_reranked += cached_results["tokens_reranked"]
                        self.total_reranking_cost += (cached_results["tokens_reranked"] / 1_000_000) * VOYAGE_RERANK_COST_PER_MILLION_TOKENS
                    
                    return cached_results["mrr"], cached_results["ndcg"]
            except Exception as e:
                logger.warning(f"Failed to load cached metrics for query {query_id}: {e}")       
        relevant_docs = self.qrels[query_id]
        all_doc_ids = list(self.corpus_embeddings.keys())
        query_norm = np.linalg.norm(query_embedding)
        query_embedding = query_embedding / query_norm if query_norm > 0 else query_embedding
        
        similarities = []
        for doc_id in all_doc_ids:
            doc_embedding = self.corpus_embeddings[doc_id]
            doc_norm = np.linalg.norm(doc_embedding)
            doc_embedding = doc_embedding / doc_norm if doc_norm > 0 else doc_embedding
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((similarity, doc_id))
        similarities.sort(reverse=True)
        initial_ranking = [(score, doc_id) for score, doc_id in similarities[:self.initial_k]]
        
        tokens_reranked = 0
        if self.use_reranker:
            docs_to_rerank = [self.corpus[doc_id] for _, doc_id in initial_ranking]
            doc_id_map = {doc: doc_id for doc, (_, doc_id) in zip(docs_to_rerank, initial_ranking)}
            reranking = self.vo.rerank(
                self.queries[query_id],
                docs_to_rerank,
                model="rerank-2",
                top_k=self.k
            )
            tokens_reranked = reranking.total_tokens
            self.total_tokens_reranked += tokens_reranked
            self.total_reranking_cost += (tokens_reranked / 1_000_000) * VOYAGE_RERANK_COST_PER_MILLION_TOKENS
            ranking = [doc_id_map[result.document] for result in reranking.results]
        else:
            ranking = [doc_id for _, doc_id in initial_ranking[:self.k]]
        mrr = 0
        for rank, doc_id in enumerate(ranking):
            if doc_id in relevant_docs:
                mrr = 1.0 / (rank + 1)
                break

        predicted_relevance = [1 if doc_id in relevant_docs else 0 for doc_id in ranking[:self.k]]
        true_relevances = [1] * len(relevant_docs)
        dcg = compute_dcg_at_k(predicted_relevance, self.k)
        idcg = compute_dcg_at_k(true_relevances, min(self.k, len(relevant_docs)))
        ndcg = dcg / idcg if idcg > 0 else 0
        cache_data = {
            "mrr": mrr,
            "ndcg": ndcg,
            "model": self.model_name,
            "dataset": self.dataset_name,
            "query_id": query_id,
            "k": self.k,
            "use_reranker": self.use_reranker,
            "initial_k": self.initial_k,
            "ranking": ranking[:self.k],
            "relevant_docs": list(relevant_docs)
        }
        
        if self.use_reranker:
            cache_data["tokens_reranked"] = tokens_reranked
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

        return mrr, ndcg
    
    def evaluate(self):
        """Run the evaluation and return metrics"""
        logger.info("Loading or generating query embeddings...")
        query_ids = [qid for qid in self.queries if qid in self.qrels and self.qrels[qid]]
        query_texts = [self.queries[qid] for qid in query_ids]
        query_embeddings = self.embed_queries(query_texts, query_ids)
        if not query_embeddings:
            logger.error("Failed to obtain query embeddings")
            return {
                f"MRR@{self.k}": 0,
                f"NDCG@{self.k}": 0,
            }
        mrr_sum = 0
        ndcg_sum = 0
        for query_id in tqdm(query_embeddings):
            mrr, ndcg = self.calculate_metrics(query_embeddings[query_id], query_id)
            mrr_sum += mrr
            ndcg_sum += ndcg
        query_count = len(query_embeddings)
        metrics = {
            f"MRR@{self.k}": mrr_sum / query_count if query_count > 0 else 0,
            f"NDCG@{self.k}": ndcg_sum / query_count if query_count > 0 else 0,
        }
        if self.use_reranker:
            metrics.update({
                "total_reranking_cost": self.total_reranking_cost,
                "total_tokens_reranked": self.total_tokens_reranked,
            })
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Calculate MRR using cached embeddings")
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--skip_datasets", type=str, default = '')
    parser.add_argument("--k", type=int, default=10,
                      help="Number of documents to consider for metrics (default: 10)")
    parser.add_argument("--endpoint", type=str, default="")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--use_reranker", action="store_true",
                      help="Whether to use Voyage reranker after initial retrieval")
    parser.add_argument("--initial_k", type=int, default=100,
                      help="Number of documents to retrieve before reranking (default: 100)")
    args = parser.parse_args()
    if args.dataset == 'all':
        datasets = list(set(DATASET_PATHS.keys()) - set(args.skip_datasets.split()))
    else:
        datasets = [args.dataset]

    all_metrics = {}
    for d in datasets:
        logger.info(f"Starting evaluation for {args.model} on {d}")
        evaluator = EmbeddingEvaluator(
            args.model, 
            d, 
            endpoint=args.endpoint, 
            k=args.k,
            use_reranker=args.use_reranker,
            initial_k=args.initial_k,
        )
        metrics = evaluator.evaluate()
        for metric_name, metric_value in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = 0
            all_metrics[metric_name] += metric_value
            logger.info(f"{metric_name}: {metric_value}")
                
        if args.save:
            model_name_safe = args.model.replace("/", "-")
            output_dict = {
                "model": args.model,
                "dataset": d,
                "k": args.k,
                "use_reranker": args.use_reranker,
                "initial_k": args.initial_k if args.use_reranker else args.k,
                "metrics": metrics
            }
            output_filename = f"performance_stats/{model_name_safe}_{d}_performance_stats.json"
            if args.use_reranker:
                output_filename = f"performance_stats/{model_name_safe}_{d}_rerank_performance_stats.json"
            with open(output_filename, 'w') as f:
                json.dump(output_dict, f, indent=2)
            logger.info(f"Metrics saved to {output_filename}")
    
    if len(datasets) > 0:
        logger.info(f"\n===== Average metrics across {len(datasets)} datasets =====")
        for metric_name, metric_sum in all_metrics.items():
            avg_value = metric_sum / len(datasets)
            if isinstance(avg_value, float):
                logger.info(f"Average {metric_name}: {avg_value:.4f}")
            else:
                logger.info(f"Average {metric_name}: {avg_value}")

if __name__ == "__main__":
    main()