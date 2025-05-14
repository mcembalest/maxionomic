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
from openai import OpenAI
from sentence_transformers import SentenceTransformer


from corpus_embedding import DATASET_PATHS

VOYAGE_RERANK_COST_PER_MILLION_TOKENS = 0.05

prompt_dict = {
    "arguana": "Instruct: Given a claim, find documents that refute the claim\nQuery: ",
    "climate-fever": "Instruct: Given a claim about climate change, retrieve documents that support or refute the claim\nQuery: ",
    "dbpedia": "Instruct: Given a query, retrieve relevant entity descriptions from DBPedia\nQuery: ",
    "fever": "Instruct: Given a claim, retrieve documents that support or refute the claim\nQuery: ",
    "fiqa": "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: ",
    "hotpot": "Instruct: Given a multi-hop question, retrieve documents that can help answer the question\nQuery: ",
    "msmarco": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "nfcorpus": "Instruct: Given a question, retrieve relevant documents that best answer the question\nQuery: ",
    "nq": "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery: ",
    "quora": "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\nQuery: ",
    "scidocs": "Instruct: Given a scientific paper title, retrieve paper abstracts that are cited by the given paper\nQuery: ",
    "scifact": "Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery: ",
    "touche": "Instruct: Given a question, retrieve detailed and persuasive arguments that answer the question\nQuery: ",
}

def add_eos(input_examples, eos_token):
  input_examples = [input_example + eos_token for input_example in input_examples]
  return input_examples


def compute_dcg_at_k(relevances, k):
    dcg = 0
    for i in range(min(len(relevances), k)):
        dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
    return dcg

class EmbeddingEvaluator:
    def __init__(self, model_name, dataset_name, endpoint="", k=10, use_reranker=False, initial_k=100, use_st=False, st_model=None):
        """Initialize the embedding evaluator"""
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json"}
        self.k = k
        self.use_reranker = use_reranker
        self.initial_k = initial_k if use_reranker else k
        self.use_sentence_transformer = use_st
        self.st_model = st_model  # Store pre-loaded ST model
        
        # Determine query prefix based on model and dataset
        if "nomic" in model_name:
            self.query_prefix="search_query: "
        elif "e5-mistral" in model_name:
            self.query_prefix = prompt_dict.get(dataset_name, "")
            if not self.query_prefix:
                logger.warning(f"Dataset {dataset_name} not found in e5_prompt_dict. Using empty query prefix.")
        elif "NV-Embed-v2" in model_name:
            self.query_prefix = prompt_dict.get(dataset_name, "")
            logger.info(f"FOUND QUERY PREFIX: {self.query_prefix}")
            if not self.query_prefix:
                logger.warning(f"Dataset {dataset_name} not found in nv_prompt_dict. Using empty query prefix.")
        else:
            self.query_prefix = ""
        
        self.total_reranking_cost = 0.0
        self.total_tokens_reranked = 0
        
        # Initialize clients/APIs as needed
        self.use_voyage = "voyage" in model_name.lower()
        self.use_nvidia_api = "nvidia/llama-3.2-nv-embedqa-1b-v2" in model_name # Specific check for Nvidia API model
        self.vo = None
        self.nvidia_client = None
        
        # Initialize Voyage client if using Voyage model OR reranker
        voyage_api_key = os.environ.get("VOYAGE_API_KEY", None)
        if use_reranker or self.use_voyage:
            if not voyage_api_key:
                raise ValueError("Voyage API key (VOYAGE_API_KEY env var) required for reranking or Voyage models")
            voyageai.api_key = voyage_api_key
            self.vo = voyageai.Client()
            logger.info("Voyage AI client initialized.")
        
        # Initialize Nvidia client if using the specific Nvidia API model AND NOT using SentenceTransformer locally
        if self.use_nvidia_api and not self.use_sentence_transformer:
            if not endpoint:
                raise ValueError("Endpoint URL required for Nvidia API model when not using SentenceTransformer locally (--use_st)")
            self.nvidia_client = OpenAI(
                api_key="not-needed", 
                base_url=self.endpoint
            )
            logger.info(f"Nvidia API client initialized for endpoint: {self.endpoint}")
        
        # Initialize SentenceTransformer model parts (tokenizer) only if using ST
        self.tokenizer = None
        if self.use_sentence_transformer:
            if self.st_model:
                self.tokenizer = self.st_model.tokenizer
                logger.info("Using pre-loaded SentenceTransformer model and tokenizer.")
            else:
                # This case should ideally not happen if main logic is correct,
                # but signifies an issue if st_model wasn't passed when use_st=True.
                raise ValueError("use_st is True, but no SentenceTransformer model was provided.")
        elif self.use_nvidia_api:
             # If using Nvidia API, might still need a tokenizer for other purposes (e.g. future token counting)
             # Use the standard one associated with the model if needed elsewhere.
             try:
                 # Let's use a default tokenizer for Nvidia API path if needed later
                 # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B") ? Decide if needed
                 pass # No tokenizer needed for current Nvidia API embedding logic
             except Exception as e:
                 logger.warning(f"Could not load default tokenizer for Nvidia model: {e}")
        else:
            # If using a generic endpoint (not ST, not Voyage, not Nvidia API), we might need a tokenizer
            # if the endpoint doesn't handle prefixing or if we need token counts later.
            # For now, assume endpoint handles everything if not ST.
             pass
        
        # Load dataset components after initializing clients/models
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
            # Ensure cached embeddings match the requested query IDs
            if set(cached_embeddings.keys()) == set(query_ids):
                logger.info("Using cached query embeddings.")
                return cached_embeddings
            else:
                logger.warning("Cached query IDs mismatch requested IDs. Recomputing.")

        query_embeddings = {}
        logger.info(f"Embedding {len(query_texts)} queries using model: {self.model_name}")

        try:
            if self.use_sentence_transformer:
                if not self.st_model or not self.tokenizer:
                    raise ValueError("SentenceTransformer model or tokenizer not available.")
                # Add EOS token if required by the model/tokenizer
                # texts_to_embed = add_eos(query_texts, self.tokenizer.eos_token)
                texts_to_embed = query_texts
                logger.debug(f"Embedding queries with ST. Example: '{texts_to_embed[0][:100]}...'")
                # Specify query prompt name if applicable for the model
                embeddings = self.st_model.encode(texts_to_embed, batch_size=len(query_texts), normalize_embeddings=True)
            elif self.use_voyage:
                if not self.vo: raise ValueError("Voyage AI client not initialized.")
                logger.debug("Embedding queries with Voyage API.")
                # Prefixing is handled by input_type='query'
                result = self.vo.embed(
                    query_texts,
                    model=self.model_name.replace("voyageai/", ""),
                    input_type="query"
                )
                embeddings = result.embeddings
            elif self.use_nvidia_api:
                if not self.nvidia_client: raise ValueError("Nvidia API client not initialized.")
                logger.debug("Embedding queries with Nvidia API.")
                # Apply prefix if needed (though input_type='query' might handle it)
                prefixed_texts = [self.query_prefix + text for text in query_texts]
                try:
                    response = self.nvidia_client.embeddings.create(
                        input=prefixed_texts, # Send prefixed text
                        model=self.model_name, # Use the actual model name from args
                        encoding_format="float",
                        extra_body={"input_type": "query", "truncate": "END"}
                    )
                    embeddings = [data.embedding for data in response.data]
                except Exception as e:
                    logger.error(f"Nvidia embedding request failed: {str(e)}")
                    return None # Indicate failure
            else:
                logger.debug(f"Embedding queries with generic endpoint: {self.endpoint}")
                if not self.endpoint:
                    raise ValueError("Endpoint URL required for generic model embedding.")

                prefixed_texts = [self.query_prefix + x for x in query_texts]
                payload = {"inputs": prefixed_texts, "normalize": True} # Add normalize=True standardly
                response = requests.post(
                    f"{self.endpoint}/embed",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                embeddings = response.json()
                # Ensure response is a list of lists (embeddings)
                if not isinstance(embeddings, list) or (embeddings and not isinstance(embeddings[0], list)):
                    raise ValueError(f"Unexpected response format from endpoint: {type(embeddings)}")

            # Check if the number of embeddings matches the number of queries
            if len(embeddings) != len(query_ids):
                raise ValueError(f"Mismatch in query count ({len(query_ids)}) and embeddings received ({len(embeddings)}). Check API response.")

            query_embeddings = {qid: emb for qid, emb in zip(query_ids, embeddings)}
            self._save_query_embeddings(query_embeddings) # Save successfully computed embeddings
            logger.info(f"Successfully embedded {len(query_embeddings)} queries.")
            return query_embeddings

        except Exception as e:
            logger.error(f"Error during query embedding: {str(e)}", exc_info=True) # Log traceback
            return None # Indicate failure
    
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
            logger.info(f"Reranking top {self.initial_k} docs for query {query_id} with Voyage...")
            reranking = self.vo.rerank(
                self.queries[query_id],
                docs_to_rerank,
                model="rerank-2",
                top_k=self.k
            )
            logger.info(f"Voyage reranking completed for query {query_id}.")
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
    parser.add_argument("--model", type=str, required=True, help="Model name for embeddings (HuggingFace name, 'voyageai/...' or 'nvidia/...')")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or 'all' to run on multiple datasets")
    parser.add_argument("--skip_datasets", type=str, default='', help="Comma-separated list of datasets to skip if --dataset=all")
    parser.add_argument("--k", type=int, default=10, help="Number of documents to consider for metrics (default: 10)")
    parser.add_argument("--endpoint", type=str, default="", help="Endpoint URL for API-based models (Nvidia or generic TGI)")
    parser.add_argument("--save", action="store_true", help="Save performance metrics to a file")
    parser.add_argument("--use_reranker", action="store_true", help="Whether to use Voyage reranker after initial retrieval")
    parser.add_argument("--initial_k", type=int, default=100, help="Number of documents to retrieve before reranking (default: 100)")
    parser.add_argument("--use_st", action="store_true", help="Force using SentenceTransformer for local model loading/embedding, even for models that might have API options (e.g., Nvidia)")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Maximum sequence length for SentenceTransformer models (default: 512)")

    args = parser.parse_args()

    # --- Pre-load SentenceTransformer model if specified ---
    st_model = None
    if args.use_st:
        logger.info(f"--- Loading SentenceTransformer Model ({args.model}) ---")
        try:
            # Consider adding device selection if needed: device='cuda'
            st_model = SentenceTransformer(args.model, trust_remote_code=True)
            st_model.max_seq_length = args.max_seq_length
            # Use half precision for efficiency, ensure compatibility if changing
            st_model = st_model.half()
            logger.info(f"SentenceTransformer model loaded successfully. Max seq length: {st_model.max_seq_length}")
            logger.info("--- Model Loaded ---")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{args.model}': {e}", exc_info=True)
            exit(1) # Exit if ST model loading fails when explicitly requested

    if args.dataset == 'all':
        datasets_to_skip = set(ds.strip() for ds in args.skip_datasets.split(',') if ds.strip())
        datasets = [d for d in DATASET_PATHS.keys() if d not in datasets_to_skip]
        logger.info(f"Running on all datasets except: {datasets_to_skip}")
    else:
        datasets = [args.dataset]
        if args.dataset not in DATASET_PATHS:
            logger.error(f"Specified dataset '{args.dataset}' is not valid. Available: {list(DATASET_PATHS.keys())}")
            exit(1)

    all_metrics_summary = {} # Store metrics per dataset
    aggregated_metrics = {} # Store sum of metrics across datasets for averaging

    for d in datasets:
        logger.info(f"\n--- Starting Evaluation: Model '{args.model}' on Dataset '{d}' ---")
        try:
            evaluator = EmbeddingEvaluator(
                model_name=args.model,
                dataset_name=d,
                endpoint=args.endpoint,
                k=args.k,
                use_reranker=args.use_reranker,
                initial_k=args.initial_k,
                use_st=args.use_st,
                st_model=st_model # Pass the pre-loaded model (or None)
            )
            metrics = evaluator.evaluate()

            # Store metrics for this dataset
            all_metrics_summary[d] = metrics
            logger.info(f"--- Results for {d} --- ")
            for metric_name, metric_value in metrics.items():
                 # Accumulate for averaging later
                if metric_name not in aggregated_metrics:
                    aggregated_metrics[metric_name] = 0
                # Handle potential None values if evaluation failed partially
                if metric_value is not None:
                     aggregated_metrics[metric_name] += metric_value

                # Log individual dataset results
                if isinstance(metric_value, float):
                     logger.info(f"{metric_name}: {metric_value:.4f}")
                else:
                     logger.info(f"{metric_name}: {metric_value}")
            logger.info("------------------------")

            if args.save:
                # Define output filename
                model_name_safe = args.model.replace("/", "-")
                rerank_part = "rerank_performance_stats" if args.use_reranker else "performance_stats"
                output_dir = "performance_stats"
                os.makedirs(output_dir, exist_ok=True)
                output_filename = f"{output_dir}/{model_name_safe}_{d}_{rerank_part}.json"

                # Prepare output data
                output_dict = {
                    "model": args.model,
                    "dataset": d,
                    "k": args.k,
                    "use_reranker": args.use_reranker,
                    "initial_k": args.initial_k if args.use_reranker else args.k, # Use initial_k if reranking
                    "use_sentence_transformer": args.use_st,
                    "metrics": metrics
                }

                # Save the JSON file
                try:
                    with open(output_filename, 'w') as f:
                        json.dump(output_dict, f, indent=2)
                    logger.info(f"Metrics successfully saved to {output_filename}")
                except IOError as e:
                     logger.error(f"Error saving metrics to {output_filename}: {e}")

        except FileNotFoundError as e:
             logger.error(f"Evaluation failed for {d}: Required cache file not found - {e}. Ensure corpus embeddings were generated first.")
             all_metrics_summary[d] = {"error": f"Cache file not found: {e}"}
        except Exception as e:
            logger.error(f"!!! Evaluation failed for Model '{args.model}' on Dataset '{d}': {e} !!!", exc_info=True)
            # Record the error in the summary
            all_metrics_summary[d] = {"error": str(e)}
            # Continue to the next dataset

    # --- Summary Across Datasets ---
    num_successful_datasets = sum(1 for d in datasets if all_metrics_summary.get(d) and 'error' not in all_metrics_summary[d])

    if num_successful_datasets > 0:
        logger.info(f"\n===== Aggregated Metrics Across {num_successful_datasets} Successful Datasets =====")
        for metric_name, metric_sum in aggregated_metrics.items():
            avg_value = metric_sum / num_successful_datasets
            if isinstance(avg_value, float):
                logger.info(f"Average {metric_name}: {avg_value:.4f}")
            else:
                 # Handle aggregated non-float values like total cost/tokens if needed
                 logger.info(f"Total {metric_name}: {metric_sum}") # Log sum for totals
                 # logger.info(f"Average {metric_name}: {avg_value}") # Log average if meaningful

        # Optionally save the overall summary if multiple datasets were run
        if len(datasets) > 1 and args.save:
            model_name_safe = args.model.replace("/", "-")
            rerank_part = f"rerank{args.k}from{args.initial_k}" if args.use_reranker else f"k{args.k}"
            summary_filename = f"performance_stats/{model_name_safe}_ALL-DATASETS_{rerank_part}_summary.json"
            summary_data = {
                 "model": args.model,
                 "datasets_run": datasets,
                 "datasets_skipped": list(datasets_to_skip) if args.dataset == 'all' else [],
                 "k": args.k,
                 "use_reranker": args.use_reranker,
                 "initial_k": args.initial_k if args.use_reranker else args.k,
                 "use_sentence_transformer": args.use_st,
                 "num_successful_datasets": num_successful_datasets,
                 "average_metrics": {
                     name: (total / num_successful_datasets)
                     for name, total in aggregated_metrics.items()
                 },
                 "per_dataset_metrics": all_metrics_summary
            }
            try:
                with open(summary_filename, 'w') as f:
                    json.dump(summary_data, f, indent=2)
                logger.info(f"Overall summary saved to {summary_filename}")
            except IOError as e:
                logger.error(f"Error saving overall summary to {summary_filename}: {e}")

    elif len(datasets) > 0:
         logger.warning("\nNo datasets completed successfully. Cannot calculate average metrics.")
    else:
         logger.info("\nNo datasets were selected to run.")

    logger.info("\n--- Evaluation Run Finished ---")

if __name__ == "__main__":
    main()