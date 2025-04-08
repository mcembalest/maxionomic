from argparse import ArgumentParser
import requests
import time
import numpy as np
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import voyageai
import concurrent.futures
from openai import OpenAI
from sentence_transformers import SentenceTransformer

DATASET_PATHS = {
    "arguana": "zeta-alpha-ai/NanoArguAna",
    "climate-fever": "zeta-alpha-ai/NanoClimateFEVER",
    "dbpedia": "zeta-alpha-ai/NanoDBPedia",
    "fever": "zeta-alpha-ai/NanoFEVER",
    "fiqa": "zeta-alpha-ai/NanoFiQA2018",
    "hotpot": "zeta-alpha-ai/NanoHotpotQA",
    "msmarco": "zeta-alpha-ai/NanoMSMARCO",
    "nfcorpus": "zeta-alpha-ai/NanoNFCorpus",
    "nq": "zeta-alpha-ai/NanoNQ",
    "quora": "zeta-alpha-ai/NanoQuoraRetrieval",
    "scidocs": "zeta-alpha-ai/NanoSciDocs",
    "scifact": "zeta-alpha-ai/NanoSciFact",
    "touche": "zeta-alpha-ai/NanoTouche2020",
}

def add_eos(input_examples, eos_token):
  input_examples = [input_example + eos_token for input_example in input_examples]
  return input_examples


def load_model_components(model_name, max_seq_length, endpoint, use_st):
    """Loads the necessary model components (tokenizer, client, etc.) once."""
    tokenizer = None
    st_model = None
    vo = None
    nvidia_client = None
    use_voyage = "voyage" in model_name.lower()
    use_nvidia_api = "nvidia/llama-3.2-nv-embedqa-1b-v2" in model_name
    use_sentence_transformer = use_st

    if use_sentence_transformer:
        print(f"Loading SentenceTransformer model: {model_name}")
        st_model = SentenceTransformer(model_name, trust_remote_code=True)
        st_model.max_seq_length = max_seq_length
        st_model = st_model.half() # Use half precision
        tokenizer = st_model.tokenizer
        print("SentenceTransformer model loaded.")
    elif use_voyage:
        print("Initializing Voyage AI client...")
        voyage_api_key = os.environ.get("VOYAGE_API_KEY", None)
        if not voyage_api_key:
            raise ValueError("Voyage API key required for Voyage models")
        voyageai.api_key = voyage_api_key
        vo = voyageai.Client()
        # Voyage doesn't expose a tokenizer easily, handle token counts via API response
        print("Voyage AI client initialized.")
    elif use_nvidia_api:
        print("Initializing Nvidia API client...")
        nvidia_client = OpenAI(
            api_key="not-needed", # Assuming API key is handled by the environment or endpoint config
            base_url=endpoint
        )
        # Use a standard tokenizer for approximation if needed for other calculations,
        # but rely on API/model for precise embedding logic.
        # Using Llama 3.2 1B tokenizer as specified in the original code for Nvidia model
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        print("Nvidia API client initialized.")
    else:
        # Assumes TGI endpoint or similar for other models
        print(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded.")

    return tokenizer, st_model, vo, nvidia_client, use_voyage, use_nvidia_api, use_sentence_transformer


class EmbeddingsBenchmark:
    def __init__(self, model_name, batch_size, max_seq_length, endpoint, dataset_name, use_cache, concurrency,
                 tokenizer, st_model, vo, nvidia_client, use_voyage, use_nvidia_api, use_sentence_transformer):
        self.cache_dir = "embeddings_cache"
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.use_cache = use_cache
        self.endpoint = endpoint
        self.concurrency = concurrency
        self.document_prefix = ""
        if "nomic" in model_name:
            self.document_prefix = "search_document: "

        # Store pre-loaded components
        self.tokenizer = tokenizer
        self.st_model = st_model
        self.vo = vo
        self.nvidia_client = nvidia_client
        self.use_voyage = use_voyage
        self.use_nvidia_api = use_nvidia_api
        self.use_sentence_transformer = use_sentence_transformer

        self.headers = {"Content-Type": "application/json"}
        self.session = requests.Session()
        model_name_safe = model_name.replace("/", "-")
        self.cache_file = f"{self.cache_dir}/{model_name_safe}_{dataset_name}_embeddings"
        os.makedirs(self.cache_dir, exist_ok=True)

    def batch_embed(self, texts: list[str]):
        """Embed a batch of texts and return embeddings with latency"""
        start_time = time.time()
        total_tokens = 0 # Initialize total_tokens

        if self.use_sentence_transformer:
            # Ensure tokenizer and st_model are available
            if not self.tokenizer or not self.st_model:
                 raise ValueError("SentenceTransformer components not loaded correctly.")
            texts = add_eos(texts, self.tokenizer.eos_token)
            embeddings = self.st_model.encode(texts, batch_size=len(texts), normalize_embeddings=True)
            # Count tokens using the model's tokenizer AFTER adding EOS potentially
            token_counts = [min(len(self.tokenizer.encode(text)), self.max_seq_length) for text in texts]
            total_tokens = sum(token_counts)
        elif self.use_voyage:
            if not self.vo:
                 raise ValueError("Voyage AI client not initialized.")
            result = self.vo.embed(
                texts,
                model=self.model_name.replace("voyageai/", ""),
                input_type="document"
            )
            embeddings = result.embeddings
            total_tokens = result.total_tokens
        elif self.use_nvidia_api:
            if not self.nvidia_client:
                raise ValueError("Nvidia API client not initialized.")
            try:
                response = self.nvidia_client.embeddings.create(
                    input=texts,
                    model=self.model_name,
                    encoding_format="float",
                    extra_body={"input_type": "passage", "truncate": "END"}
                )
                embeddings = [data.embedding for data in response.data]
                # Approximate token count if tokenizer is available
                if self.tokenizer:
                    total_counts = [len(self.tokenizer.encode(text)) for text in texts]
                    total_tokens = sum(total_counts)
                else:
                    # Cannot estimate tokens if tokenizer wasn't loaded (e.g., if not specified for Nvidia)
                    total_tokens = None # Indicate unknown token count
            except Exception as e:
                raise Exception(f"Nvidia embedding request failed: {str(e)}")
        else: 
            if not self.tokenizer:
                 raise ValueError("Tokenizer not loaded for generic endpoint.")
            prefixed_texts = [self.document_prefix + x for x in texts]
            response = self.session.post(
                f"{self.endpoint}/embed",
                headers=self.headers,
                json={"inputs": prefixed_texts}
            )
            if response.status_code != 200:
                raise Exception(f"Embedding request failed with status {response.status_code}: {response.text}")
            embeddings = response.json()
            token_counts = self.tokenizer(
                prefixed_texts,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_length=True
            )["length"]
            total_tokens = sum(token_counts)

        latency = time.time() - start_time
        if len(embeddings) != len(texts):
            raise Exception(f"Expected {len(texts)} embeddings but got {len(embeddings)}")
        return embeddings, latency, total_tokens

    def embed_corpus(self, corpus):
        """Embed an entire corpus with batching, concurrency, and caching"""
        assert len(corpus) > 0, "Corpus is empty"
        initial_corpus_size = len(corpus)
        # Check cache using the specific cache_file path for this dataset instance
        cache_path_npy = self.cache_file + '.npy'
        cache_path_ids = self.cache_file + '_ids.json'

        if self.use_cache and os.path.exists(cache_path_npy) and os.path.exists(cache_path_ids):
            print(f"Loading embeddings from cache: {self.cache_file}")
            try:
                embeddings_array = np.load(cache_path_npy)
                with open(cache_path_ids, 'r') as f:
                    doc_ids_ordered = json.load(f)
                # Verify cache integrity
                if len(doc_ids_ordered) == embeddings_array.shape[0]:
                     all_embeddings = {doc_id: embeddings_array[i].tolist() for i, doc_id in enumerate(doc_ids_ordered)}
                     # Optional: Check if cached IDs match current corpus keys if necessary
                     # if set(doc_ids_ordered) == set(corpus.keys()):
                     if len(all_embeddings) == initial_corpus_size: # Check if size matches expected
                           print("Cache loaded successfully.")
                           return all_embeddings, 0, 0 # Return cached data, 0 duration/tokens
                     else:
                           print(f"Cache size mismatch: expected {initial_corpus_size}, got {len(all_embeddings)}. Recomputing.")
                else:
                    print(f"Cache corrupted (ID count {len(doc_ids_ordered)} != embedding count {embeddings_array.shape[0]}). Recomputing.")
            except Exception as e:
                print(f"Failed to load cache ({e}). Recomputing.")
                # Clean up potentially corrupted cache files
                # if os.path.exists(cache_path_npy): os.remove(cache_path_npy)
                # if os.path.exists(cache_path_ids): os.remove(cache_path_ids)

        # Proceed with embedding if cache is not used or loading failed/invalid
        all_embeddings = {}
        all_latencies = []
        total_tokens_processed = 0 # Renamed to avoid conflict
        doc_ids = list(corpus.keys())
        total_start_time = time.time()
        batches = []
        for i in range(0, len(doc_ids), self.batch_size):
            batch_ids = doc_ids[i:i + self.batch_size]
            batch_texts = [corpus[doc_id] for doc_id in batch_ids]
            batches.append((batch_ids, batch_texts))

        print(f"Processing {len(batches)} batches with concurrency {self.concurrency}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_batch_ids = {
                executor.submit(self.batch_embed, batch_texts): batch_ids
                for batch_ids, batch_texts in batches
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_batch_ids), total=len(batches)):
                batch_ids = future_to_batch_ids[future]
                try:
                    embeddings, latency, batch_tokens = future.result()
                    all_latencies.append(latency)
                    if batch_tokens != -1: # Only add if tokens were calculable
                       total_tokens_processed += batch_tokens
                    all_embeddings.update(dict(zip(batch_ids, embeddings)))
                except Exception as e:
                    print(f"Error processing batch for IDs {batch_ids[:5]}...: {str(e)}")
                    # Decide if to raise or continue; raising is safer
                    raise e

        total_duration = time.time() - total_start_time
        if len(all_embeddings) != initial_corpus_size:
             print(f"Warning: Final embedding count mismatch: got {len(all_embeddings)}, expected {initial_corpus_size}. Some batches may have failed.")
             # Depending on requirements, you might want to raise an error here or handle partial results

        # Save to cache if computed and caching enabled
        if self.use_cache and self.cache_file and len(all_embeddings) == initial_corpus_size: # Only save if complete
            print(f"Saving embeddings to cache: {self.cache_file}")
            try:
                doc_ids_ordered = list(all_embeddings.keys()) # Ensure consistent order
                embeddings_array = np.array([all_embeddings[doc_id] for doc_id in doc_ids_ordered])
                np.save(cache_path_npy, embeddings_array)
                with open(cache_path_ids, 'w') as f:
                    json.dump(doc_ids_ordered, f)
                print("Cache saved successfully.")
            except Exception as e:
                print(f"Failed to save cache: {e}")


        avg_latency = np.mean(all_latencies) if all_latencies else 0
        tokens_per_second = total_tokens_processed / total_duration if total_duration > 0 else 0

        print(f"Average batch latency: {avg_latency:.4f} seconds")
        print(f"Total corpus embedding duration: {total_duration:.4f} seconds")
        print(f"Total documents embedded: {len(all_embeddings)}")
        if total_tokens_processed >= 0:
            print(f"Total tokens processed: {total_tokens_processed}")
            print(f"Tokens per second: {tokens_per_second:.2f}")
        else:
            print("Total tokens processed: Not Available (e.g., Nvidia API without tokenizer)")

        return all_embeddings, total_duration, total_tokens_processed

def load_dataset_corpus(dataset_name):
    """Load corpus from a NanoBEIR dataset"""
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_PATHS.keys())}")
    dataset_path = DATASET_PATHS[dataset_name]
    corpus = load_dataset(dataset_path, "corpus", split="train")
    processed_corpus = {
        str(sample["_id"]): sample["text"] 
        for sample in corpus 
        if len(sample["text"].strip()) > 0
    }
    print(f"Loaded {len(processed_corpus)} documents from {dataset_name} corpus")
    return processed_corpus

if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark embedding models on NanoBEIR datasets")
    parser.add_argument("--model", type=str, required=True,
                      help="Name of the embedding model to use")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Dataset name or 'all' to run on all datasets")
    parser.add_argument("--endpoint", type=str, required=True,
                      help="Endpoint URL for the embedding service")
    parser.add_argument("--batch_size", type=int, default=512,
                      help="Batch size for embedding requests (default: 512)")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max_seq_length", type=int, default=8192,
                      help="Maximum sequence length (default: 8192)")
    parser.add_argument("--no_cache", action="store_false", dest="use_cache", 
                      help="Disable cache for embeddings (default: cache enabled)")
    parser.add_argument("--save", action="store_true",
                      help="Save benchmark results to file")
    parser.add_argument("--use_st", action="store_true",
                      help="Use SentenceTransformer for embeddings")
    args = parser.parse_args()

    # Load model components ONCE before the loop
    print("--- Loading Model Components ---")
    try:
        model_components = load_model_components(
            args.model, args.max_seq_length, args.endpoint, args.use_st
        )
        tokenizer, st_model, vo, nvidia_client, use_voyage, use_nvidia_api, use_sentence_transformer = model_components
        print("--- Model Components Loaded ---")
    except Exception as e:
        print(f"Failed to load model components: {e}")
        exit(1) # Exit if model loading fails

    if args.dataset == 'all':
        datasets = list(DATASET_PATHS.keys())
    else:
        datasets = [args.dataset]

    all_results = [] # Store results for all datasets

    for d in datasets:
        print(f"\n--- Starting Benchmark for Dataset: {d} ---")
        print(f"Model: {args.model}")
        print(f"Configuration: batch_size={args.batch_size}, concurrency={args.concurrency}, max_seq_length={args.max_seq_length}")
        print(f"Cache usage: {'Enabled' if args.use_cache else 'Disabled'}")
        print(f"Save results: {'Yes' if args.save else 'No'}")

        try:
            corpus = load_dataset_corpus(d)
            # Instantiate Benchmark with pre-loaded components
            benchmark = EmbeddingsBenchmark(
                model_name=args.model,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                endpoint=args.endpoint,
                dataset_name=d,
                use_cache=args.use_cache,
                concurrency=args.concurrency,
                # Pass pre-loaded components
                tokenizer=tokenizer,
                st_model=st_model,
                vo=vo,
                nvidia_client=nvidia_client,
                use_voyage=use_voyage,
                use_nvidia_api=use_nvidia_api,
                use_sentence_transformer=use_sentence_transformer
            )

            corpus_embeddings, total_duration, total_tokens = benchmark.embed_corpus(corpus)

            model_name_safe = args.model.replace("/", "-")
            output_dict = {
                "dataset": d,
                "model": args.model,
                "batch_size": args.batch_size,
                "concurrency": args.concurrency,
                "max_seq_length": args.max_seq_length,
                "total_documents": len(corpus_embeddings),
                "total_duration_seconds": round(total_duration, 4),
                "total_tokens": total_tokens if total_tokens >= 0 else "N/A",
                "tokens_per_second": round(total_tokens / total_duration, 2) if total_duration > 0 and total_tokens >=0 else "N/A",
                "error": None
            }
            all_results.append(output_dict)
            print("--- Benchmark Result ---")
            print(json.dumps(output_dict, indent=2))
            print("------------------------")

            if args.save:
                os.makedirs("latency_stats", exist_ok=True)
                output_filename = f"latency_stats/{model_name_safe}_{d}_latency_stats.json"
                try:
                    with open(output_filename, 'w') as f:
                        json.dump(output_dict, f, indent=2)
                    print(f"Results saved to {output_filename}")
                except IOError as e:
                    print(f"Error saving results to {output_filename}: {e}")

        except Exception as e:
             print(f"!!! Error benchmarking {args.model} on {d}: {e} !!!")
             error_result = {
                 "dataset": d, "model": args.model, "error": str(e),
                 # Include other args for context
                 "batch_size": args.batch_size, "concurrency": args.concurrency, "max_seq_length": args.max_seq_length,
             }
             all_results.append(error_result)
             # Decide if you want to continue to the next dataset or stop
             # continue

    # Optional: Save a summary of all results if multiple datasets were run
    if len(datasets) > 1 and args.save:
        summary_filename = f"latency_stats/{model_name_safe}_all_datasets_summary.json"
        try:
            with open(summary_filename, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSummary results saved to {summary_filename}")
        except IOError as e:
            print(f"Error saving summary results to {summary_filename}: {e}")

    print("\n--- Benchmark Run Finished ---")
