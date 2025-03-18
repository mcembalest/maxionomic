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
from transformers import AutoTokenizer


DATASET_PATHS = {
    "msmarco": "zeta-alpha-ai/NanoMSMARCO",
    "quora": "zeta-alpha-ai/NanoQuoraRetrieval",
    "nfcorpus": "zeta-alpha-ai/NanoNFCorpus",
    "fiqa": "zeta-alpha-ai/NanoFiQA",
    "scifact": "zeta-alpha-ai/NanoSciFact",
    "arguana": "zeta-alpha-ai/NanoArguAna",
    "scidocs": "zeta-alpha-ai/NanoSciDocs",
    "fever": "zeta-alpha-ai/NanoFEVER",
    "climate-fever": "zeta-alpha-ai/NanoClimateFEVER",
    "dbpedia": "zeta-alpha-ai/NanoDBPedia"
}

VOYAGE_RERANK_COST_PER_MILLION_TOKENS = 0.05

class EmbeddingEvaluator:
    def __init__(self, model_name, dataset_name, endpoint="http://18.216.76.107:8080", batch_size=10, k=10, 
                 use_reranker=False, initial_k=100):
        """Initialize the embedding evaluator"""
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json"}
        self.batch_size = batch_size
        self.k = k
        self.use_reranker = use_reranker
        self.initial_k = initial_k if use_reranker else k
        
        self.total_reranking_cost = 0.0
        self.total_tokens_reranked = 0
        self.rerank_tokenizer = AutoTokenizer.from_pretrained("voyageai/rerank-2")
        
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
                    input_type="document"
                )
                embeddings = result.embeddings
            else:
                response = requests.post(
                    f"{self.endpoint}/embed",
                    headers=self.headers,
                    json={"inputs": query_texts}
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
            
            # Count tokens using the Voyage tokenizer
            query_tokens = len(self.rerank_tokenizer.encode(self.queries[query_id], add_special_tokens=True))
            docs_tokens = sum(len(self.rerank_tokenizer.encode(doc, add_special_tokens=True)) for doc in docs_to_rerank)
            total_tokens = (query_tokens * len(docs_to_rerank)) + docs_tokens
            
            reranking = self.vo.rerank(
                self.queries[query_id],
                docs_to_rerank,
                model="rerank-2",
                top_k=self.k
            )
            
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
        return mrr
    
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
            }
        mrr_sum = 0
        for query_id in tqdm(query_embeddings):
            mrr = self.calculate_metrics(query_embeddings[query_id], query_id)
            mrr_sum += mrr
        query_count = len(query_embeddings)
        metrics = {
            f"MRR@{self.k}": mrr_sum / query_count if query_count > 0 else 0,
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
    parser.add_argument("--dataset", type=str, default="msmarco",
                      help=f"Dataset name (default: msmarco). Available: {', '.join(DATASET_PATHS.keys())}")
    parser.add_argument("--k", type=int, default=10,
                      help="Number of documents to consider for metrics (default: 10)")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--endpoint", type=str, default="http://18.216.76.107:8080",
                      help="Embedding API endpoint (default: http://18.216.76.107:8080)")
    parser.add_argument("--output", type=str, default=None,
                      help="Output file path for metrics (default: None - prints to console)")
    parser.add_argument("--use_reranker", action="store_true",
                      help="Whether to use Voyage reranker after initial retrieval")
    parser.add_argument("--initial_k", type=int, default=100,
                      help="Number of documents to retrieve before reranking (default: 100)")
    args = parser.parse_args()
    
    logger.info(f"Starting evaluation for {args.model} on {args.dataset}")
    evaluator = EmbeddingEvaluator(
        args.model, 
        args.dataset, 
        endpoint=args.endpoint, 
        batch_size=args.batch_size,
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
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "model": args.model,
                "dataset": args.dataset,
                "k": args.k,
                "use_reranker": args.use_reranker,
                "initial_k": args.initial_k if args.use_reranker else args.k,
                "metrics": metrics
            }, f, indent=2)
        logger.info(f"Metrics saved to {args.output}")

if __name__ == "__main__":
    main()