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

class EmbeddingsBenchmark:
    def __init__(self, model_name, batch_size, max_seq_length, endpoint="", dataset_name="", use_cache=False, concurrency=5):
        self.cache_dir = "embeddings_cache"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.use_cache = use_cache
        self.endpoint = endpoint
        self.concurrency = concurrency
        self.document_prefix = ""
        if "nomic" in model_name:
            self.document_prefix = "search_document: "
        self.headers = {"Content-Type": "application/json"}
        self.session = requests.Session()
        model_name_safe = model_name.replace("/", "-")
        self.cache_file = f"{self.cache_dir}/{model_name_safe}_{dataset_name}_embeddings"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.use_voyage = "voyage" in model_name.lower()
        if self.use_voyage:
            voyage_api_key = os.environ.get("VOYAGE_API_KEY", None)
            if not voyage_api_key:
                raise ValueError("Voyage API key required for Voyage models")
            voyageai.api_key = voyage_api_key
            self.vo = voyageai.Client()
    
    def batch_embed(self, texts: list[str]):
        """Embed a batch of texts and return embeddings with latency"""
        start_time = time.time()
        if self.use_voyage:
            result = self.vo.embed(
                texts,
                model=self.model_name.replace("voyageai/", ""), 
                input_type="document"
            )
            embeddings = result.embeddings
            total_tokens = result.total_tokens
        else:
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
        if self.use_cache and self.cache_file and os.path.exists(self.cache_file + '.npy') and os.path.exists(self.cache_file + '_ids.json'):
            print(f"Loading embeddings from cache: {self.cache_file}")
            embeddings_array = np.load(self.cache_file + '.npy')
            with open(self.cache_file + '_ids.json', 'r') as f:
                doc_ids_ordered = json.load(f)
            all_embeddings = {doc_id: embeddings_array[i].tolist() for i, doc_id in enumerate(doc_ids_ordered)}
            assert len(all_embeddings) == initial_corpus_size, f"Cache size mismatch: expected {initial_corpus_size}, got {len(all_embeddings)}"
            return all_embeddings, 0, 0
        all_embeddings = {}
        all_latencies = []
        total_tokens = 0
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
                    total_tokens += batch_tokens
                    all_embeddings.update(dict(zip(batch_ids, embeddings)))
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    raise e
        total_duration = time.time() - total_start_time
        assert len(all_embeddings) == initial_corpus_size, f"Final embedding count mismatch: got {len(all_embeddings)}, expected {initial_corpus_size}"
        if self.use_cache and self.cache_file:
            print(f"Saving embeddings to cache: {self.cache_file}")
            doc_ids_ordered = list(all_embeddings.keys())
            embeddings_array = np.array([all_embeddings[doc_id] for doc_id in doc_ids_ordered])
            np.save(self.cache_file + '.npy', embeddings_array)
            with open(self.cache_file + '_ids.json', 'w') as f:
                json.dump(doc_ids_ordered, f)
        avg_latency = np.mean(all_latencies)
        tokens_per_second = total_tokens / total_duration
        print(f"Average batch latency: {avg_latency:.4f} seconds")
        print(f"Total corpus embedding duration: {total_duration:.4f} seconds")
        print(f"Total documents embedded: {len(all_embeddings)}")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        return all_embeddings, total_duration, total_tokens

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
    args = parser.parse_args()
    if args.dataset == 'all':
        datasets = list(DATASET_PATHS.keys())
    else:
        datasets = [args.dataset]
    for d in datasets:
        print(f"Benchmark starting for {args.model} on {d}")
        print(f"Configuration: batch_size={args.batch_size}, concurrency={args.concurrency}")
        print(f"Cache usage: {'Enabled' if args.use_cache else 'Disabled'}")
        print(f"Save results: {'Yes' if args.save else 'No'}")
        corpus = load_dataset_corpus(d)
        benchmark = EmbeddingsBenchmark(
            args.model,
            args.batch_size,
            args.max_seq_length,
            endpoint=args.endpoint,
            dataset_name=d,
            use_cache=args.use_cache,
            concurrency=args.concurrency
        )
        corpus_embeddings, total_duration, total_tokens = benchmark.embed_corpus(corpus)
        model_name_safe = args.model.replace("/", "-")
        output_dict = {
            "dataset": d,
            "model": args.model,
            "batch_size": args.batch_size,
            "concurrency": args.concurrency,
            "max_seq_length": args.max_seq_length,
            "total_duration_seconds": total_duration,
            "total_tokens": total_tokens,
            "tokens_per_second": total_tokens / total_duration if total_duration > 0 else 0,
        }
        print(output_dict)
        if args.save:
            os.makedirs("latency_stats", exist_ok=True)
            with open(f"latency_stats/{model_name_safe}_{d}_latency_stats.json", 'w') as f:
                json.dump(output_dict, f, indent=2)
