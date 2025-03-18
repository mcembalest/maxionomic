import argparse
from datasets import load_dataset
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

def truncate_text(tokenizer, text, max_length):
    """Match the encode-decode process from benchmark_corpus_embedding"""
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)

def analyze_dataset_truncation(dataset_name, model_name, max_seq_length):
    """Analyze how many documents would be truncated in the dataset."""
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_PATHS.keys())}")
    dataset_path = DATASET_PATHS[dataset_name]
    corpus = load_dataset(dataset_path, "corpus", split="train")
    processed_corpus = {
        str(sample["_id"]): sample["text"] 
        for sample in corpus 
        if len(sample["text"].strip()) > 0
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    total_docs = 0
    truncated_docs = 0
    total_tokens = 0
    max_tokens_seen = 0
    max_tokens_doc_id = None
    token_lengths = []
    for doc_id, text in processed_corpus.items():
        total_docs += 1
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_count = len(tokens)
        total_tokens += token_count
        token_lengths.append(token_count)
        if token_count > max_tokens_seen:
            max_tokens_seen = token_count
            max_tokens_doc_id = doc_id
            
        if token_count > max_seq_length:
            truncated_docs += 1
    
    print(f"\nDataset Metadata for {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Total documents: {total_docs}")
    print(f"Total tokens across corpus: {total_tokens:,}")
    print(f"Average tokens per document: {total_tokens/total_docs:.1f}")
    print(f"Documents requiring truncation: {truncated_docs} ({(truncated_docs/total_docs)*100:.2f}%)")
    print(f"\nDebug info:")
    print(f"Maximum tokens in any document: {max_tokens_seen}")
    print(f"ID of document with max tokens: {max_tokens_doc_id}")
    print(f"Token length distribution:")
    print(f"  Min: {min(token_lengths)}")
    print(f"  Max: {max(token_lengths)}")
    print(f"  Mean: {sum(token_lengths)/len(token_lengths):.1f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze dataset truncation metadata")
    parser.add_argument("--dataset", type=str, required=True,
                      help=f"Dataset name. Available: {', '.join(DATASET_PATHS.keys())}")
    parser.add_argument("--model", type=str, required=True,
                      help="Model name to use for tokenization (e.g. bert-base-uncased)")
    parser.add_argument("--max_seq_length", type=int, required=True,
                      help="Maximum sequence length to analyze truncation against")
    
    args = parser.parse_args()
    analyze_dataset_truncation(args.dataset, args.model, args.max_seq_length)

if __name__ == "__main__":
    main()
