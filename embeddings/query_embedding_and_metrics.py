import argparse
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

def get_prefix(dataset_name: str) -> str:
    
    raise ValueError(f"unrecognized dataset: {dataset_name}")

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
            
            # Process relevance judgments - simplified to store single relevant doc per query
            processed_qrels = {}
            for item in qrels:
                query_id = str(item["query-id"])
                processed_qrels[query_id] = str(item["corpus-id"])
            
            missing_docs = set(doc_id for doc_id in processed_qrels.values() 
                              if doc_id not in processed_corpus)
            if missing_docs:
                logger.warning(f"Found {len(missing_docs)} referenced documents that don't exist in corpus")
                # Remove queries whose relevant document is missing
                processed_qrels = {qid: doc_id for qid, doc_id in processed_qrels.items() 
                                 if doc_id not in missing_docs}
            
            logger.info(f"Dataset loaded: {len(processed_corpus)} docs, {len(processed_queries)} queries")
            return processed_corpus, processed_queries, processed_qrels
            
        except Exception as e:
            raise ValueError(f"Error loading dataset {self.dataset_name}: {str(e)}")
    
    def _load_cached_embeddings(self):
        """Load cached embeddings from the cache directory"""
        cache_dir = "embeddings_cache"
        model_name_safe = self.model_name.replace("/", "-")

        # todo: dont hard code sequence length in filename (e.g. s8192)
        cache_file = f"{cache_dir}/{model_name_safe}_{self.dataset_name}_s8192_embeddings"
        
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
        cache_file = f"{cache_dir}/{model_name_safe}_{self.dataset_name}_queries_s8192"
        
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
        cache_file = f"{cache_dir}/{model_name_safe}_{self.dataset_name}_queries_s8192"
        
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
                # note: we used consistent prefix for nomic server-side, 
                # but benchmark-dataset-specific prefixes are needed for e5-mistral
                payload = {"inputs": query_texts}
                if "e5-mistral" in self.model_name:
                    payload = {
                        "inputs": [
                            e5_mistral_prompt_dict[self.dataset_name] + x
                            for x in query_texts
                        ]
                    }
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
    
    def calculate_metrics(self, query_embedding, query_id):
        """Calculate metrics for a single query"""
        relevant_doc = self.qrels[query_id]  # Now just a single document ID
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
        
        if self.use_reranker:
            docs_to_rerank = [self.corpus[doc_id] for _, doc_id in initial_ranking]
            doc_id_map = {doc: doc_id for doc, (_, doc_id) in zip(docs_to_rerank, initial_ranking)}
            
            reranking = self.vo.rerank(
                self.queries[query_id],
                docs_to_rerank,
                model="rerank-2",
                top_k=self.k
            )

            total_tokens = reranking.total_tokens
            
            self.total_tokens_reranked += total_tokens
            self.total_reranking_cost += (total_tokens / 1_000_000) * VOYAGE_RERANK_COST_PER_MILLION_TOKENS
            
            # Extract doc_ids in order of relevance scores
            ranking = [doc_id_map[result.document] for result in reranking.results]
        else:
            ranking = [doc_id for _, doc_id in initial_ranking[:self.k]]

        # Calculate MRR@k
        mrr = 0
        for rank, doc_id in enumerate(ranking):
            if doc_id == relevant_doc:
                mrr = 1.0 / (rank + 1)
                break

        # Calculate NDCG@k
        # For binary relevance, NDCG = 1/log2(r+1) where r is the rank of the relevant doc
        ndcg = 0
        for rank, doc_id in enumerate(ranking):
            if doc_id == relevant_doc:
                ndcg = 1.0 / np.log2(rank + 2)
                break

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
        
        # Add reranking cost metrics if reranker was used
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
            if isinstance(metric_value, float):
                logger.info(f"{metric_name}: {metric_value:.4f}")
            else:
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

if __name__ == "__main__":
    main()