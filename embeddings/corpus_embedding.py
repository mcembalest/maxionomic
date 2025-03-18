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

class EmbeddingsBenchmark:
    def __init__(self, model_name, batch_size, max_seq_length, endpoint="http://18.216.76.107:8080", dataset_name="", use_cache=False):
        self.cache_dir = "embeddings_cache"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.use_cache = use_cache
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json"}
        model_name_safe = model_name.replace("/", "-")
        self.cache_file = f"{self.cache_dir}/{model_name_safe}_{dataset_name}_s{max_seq_length}_embeddings"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.use_voyage = "voyage" in model_name.lower()
        if self.use_voyage:
            voyage_api_key = os.environ.get("VOYAGE_API_KEY", None)
            if not voyage_api_key:
                raise ValueError("Voyage API key required for Voyage models")
            voyageai.api_key = voyage_api_key
            self.vo = voyageai.Client()
    
    def batch_embed(self, texts):
        """Embed a batch of texts and return embeddings with latency"""
        start_time = time.time()
        token_counts = [
            len(self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_seq_length))
            for text in texts
        ]
        
        if self.use_voyage:
            result = self.vo.embed(
                texts,
                model=self.model_name.replace("voyageai/", ""), 
                input_type="document"
            )
            embeddings = result.embeddings
        else:
            response = requests.post(
                f"{self.endpoint}/embed",
                headers=self.headers,
                json={"inputs": texts}
            )
            if response.status_code != 200:
                raise Exception(f"Embedding request failed with status {response.status_code}: {response.text}")
            embeddings = response.json()
        
        latency = time.time() - start_time
        if len(embeddings) != len(texts):
            raise Exception(f"Expected {len(texts)} embeddings but got {len(embeddings)}")
        return embeddings, latency, sum(token_counts)

    def embed_corpus(self, corpus):
        """Embed an entire corpus with batching and caching"""
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
        total_docs_processed = 0
        
        for i in tqdm(range(0, len(doc_ids), self.batch_size)):
            batch_ids = doc_ids[i:i + self.batch_size]
            if i + self.batch_size <= len(doc_ids):
                assert len(batch_ids) == self.batch_size, f"Unexpected batch size: {len(batch_ids)}"
            
            batch_texts = [corpus[doc_id] for doc_id in batch_ids]
            assert len(batch_texts) == len(batch_ids), "Batch texts and IDs size mismatch"
            
            embeddings, latency, batch_tokens = self.batch_embed(batch_texts)
            all_latencies.append(latency)
            total_tokens += batch_tokens
            
            assert len(embeddings) == len(batch_texts), f"Embedding count mismatch: expected {len(batch_texts)}, got {len(embeddings)}"
            
            for doc_id, embedding in zip(batch_ids, embeddings):
                all_embeddings[doc_id] = embedding
                total_docs_processed += 1
        
        total_duration = time.time() - total_start_time
        
        assert total_docs_processed == initial_corpus_size, f"Document count mismatch: processed {total_docs_processed}, expected {initial_corpus_size}"
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
    parser.add_argument("--model", type=str, help="For file naming only, model is loaded in EC2 instance")
    parser.add_argument("--dataset", type=str, default="msmarco",
                      help=f"Dataset name (default: msmarco). Available: {', '.join(DATASET_PATHS.keys())}")
    parser.add_argument("--endpoint", type=str, default="http://18.216.76.107:8080")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--no_cache", action="store_false", dest="use_cache", 
                      help="Disable cache for embeddings (default: cache enabled)")
    args = parser.parse_args()
    
    print(f"Benchmark starting for {args.model} on {args.dataset}")
    print(f"Cache usage: {'Enabled' if args.use_cache else 'Disabled'}")

    corpus = load_dataset_corpus(args.dataset)
    benchmark = EmbeddingsBenchmark(
        args.model,
        args.batch_size,
        args.max_seq_length,
        endpoint=args.endpoint,
        dataset_name=args.dataset,
        use_cache=args.use_cache
    )
    corpus_embeddings, total_duration, total_tokens = benchmark.embed_corpus(corpus)
    
    model_name_safe = args.model.replace("/", "-")
    with open(f"latency_stats/{model_name_safe}_{args.dataset}_b{args.batch_size}_s{args.max_seq_length}_latency_stats.json", 'w') as f:
        json.dump({
            "dataset": args.dataset,
            "model": args.model,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "total_duration_seconds": total_duration,
            "total_tokens": total_tokens,
            "tokens_per_second": total_tokens / total_duration if total_duration > 0 else 0,
        }, f, indent=2)
