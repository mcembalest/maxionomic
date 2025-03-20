import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator

model_name = 'nomic-ai/nomic-embed-text-v1.5'
model_name_safe = model_name.replace("/", "-")
model = SentenceTransformer(
    model_name, trust_remote_code=True, device='cuda'
)
query_prompts = {
    "climatefever": "search_query: ",
    "dbpedia": "search_query: ",
    "fever": "search_query: ",
    "fiqa2018": "search_query: ",
    "hotpotqa": "search_query: ",
    "msmarco": "search_query: ",
    "nfcorpus": "search_query: ",
    "nq": "search_query: ",
    "quoraretrieval": "search_query: ",
    "scidocs": "search_query: ",
    "arguana": "search_query: ",
    "scifact": "search_query: ",
    "touche2020": "search_query: ",
}
corpus_prompts = {
    "climatefever": "search_document: ",
    "dbpedia": "search_document: ",
    "fever": "search_document: ",
    "fiqa2018": "search_document: ",
    "hotpotqa": "search_document: ",
    "msmarco": "search_document: ",
    "nfcorpus": "search_document: ",
    "nq": "search_document: ",
    "quoraretrieval": "search_document: ",
    "scidocs": "search_document: ",
    "arguana": "search_document: ",
    "scifact": "search_document: ",
    "touche2020": "search_document: ",
}
evaluator = NanoBEIREvaluator(query_prompts=query_prompts,corpus_prompts=corpus_prompts)
results = evaluator(model)
with open(f"{model_name_safe}_nanobeir_results.json", 'w') as f:
    json.dump(results, f, indent=2)