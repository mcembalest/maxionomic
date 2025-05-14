from mcp.server.fastmcp import FastMCP, Context
from nomic import atlas, AtlasDataset
import pandas as pd
import os
from typing import List, Dict, Optional, Union, Any
import requests

mcp = FastMCP("Nomic Atlas Tools", dependencies=["nomic", "pandas", "requests"])

@mcp.resource("config://api_key")
def get_api_key() -> str:
    """Get the Nomic API key configuration"""
    api_key = os.environ.get("NOMIC_API_KEY")
    return "API key not found. Please set the NOMIC_API_KEY environment variable." if not api_key else f"Nomic API key is configured: {api_key[:4]}...{api_key[-4:]}"

@mcp.tool()
def list_datasets() -> List[Dict[str, str]]:
    """List all datasets accessible to the current user in Nomic Atlas"""
    api_key = os.environ.get("NOMIC_API_KEY")
    if not api_key:
        return [{"error": "API key not found. Please set the NOMIC_API_KEY environment variable."}]
    
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {api_key}'}
    response = requests.request("GET", "https://api-atlas.nomic.ai/v1/organization/nomic", headers=headers, data={})
    
    if response.status_code != 200:
        return [{"error": f"API request failed with status code {response.status_code}"}]
    
    return [{"name": p["project_name"], "total_datums_in_project": p.get("total_datums_in_project", 0)} 
            for p in response.json().get("projects", [])]

@mcp.tool()
def upload_dataset(data_path: str, dataset_name: str, text_field: str) -> str:
    """
    Upload a dataset to Nomic Atlas

    You need to choose a name for the dataset and choose one of the fields from the dataset as the text_field
    
    Args:
        data_path: Path to the CSV or JSON file
        dataset_name: Name for the dataset in Atlas
        text_field: Column name containing text to embed
    """
    if not data_path.endswith(('.csv', '.json')):
        return "Unsupported file format. Please use CSV or JSON files."
    
    df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_json(data_path)
    atlas.map_data(data=df, indexed_field=text_field, identifier=dataset_name)
    return f"Successfully created dataset '{dataset_name}' with {len(df)} records"

@mcp.tool()
def vector_search(dataset_identifier: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform vector search on a dataset

    dataset_identifier must be all lowercase no spaces (with a dash - for every space)
    
    Args:
        dataset_identifier: The project name from list_datasets (e.g. 'worldbank-project-documents-multilingual')
        query: The search query text
        k: Number of results to return
    """
    api_key = os.environ.get("NOMIC_API_KEY")
    if not api_key:
        return [{"error": "API key not found. Please set the NOMIC_API_KEY environment variable."}]

    atlas_dataset = AtlasDataset(dataset_identifier)
    payload = {"projection_id": atlas_dataset.maps[0].projection_id, "k": k, "query": query}
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {api_key}'}
    
    response = requests.request("POST", "https://api-atlas.nomic.ai/v1/query/topk", headers=headers, json=payload)
    return [{"error": f"Search request failed with status code {response.status_code}"}] if response.status_code != 200 else response.json()

@mcp.tool()
def add_data_to_dataset(dataset_identifier: str, data_path: str) -> str:
    """
    Add data to an existing Atlas dataset

    dataset_identifier must be all lowercase no spaces (with a dash - for every space)
    
    Args:
        dataset_identifier: The identifier for the dataset (org/dataset format)
        data_path: Path to the CSV or JSON file with new data
    """
    if not data_path.endswith(('.csv', '.json')):
        return "Unsupported file format. Please use CSV or JSON files."
    
    df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_json(data_path)
    AtlasDataset(dataset_identifier).add_data(data=df)
    return f"Successfully added {len(df)} records to dataset '{dataset_identifier}'"

@mcp.tool()
def create_index(
    dataset_identifier: str, 
    indexed_field: str, 
    name: Optional[str] = None,
    build_topic_model: bool = True,
    detect_duplicates: bool = True,
    duplicate_cutoff: float = 0.95
) -> str:
    """
    Create a new index/map for a dataset

    dataset_identifier must be all lowercase no spaces (with a dash - for every space)
    
    Args:
        dataset_identifier: The identifier for the dataset (org/dataset format)
        indexed_field: Field to use for indexing/embedding
        name: Optional name for the index
        build_topic_model: Whether to build a topic model
        detect_duplicates: Whether to detect duplicate entries
        duplicate_cutoff: Threshold for duplicate detection (0-1)
    """
    dataset = AtlasDataset(dataset_identifier)
    
    map_name = name or f"{dataset.name}_map"
    
    # Create the index with topic modeling and duplicate detection
    dataset.create_index(
        name=map_name,
        indexed_field=indexed_field,
        topic_model={
            "build_topic_model": build_topic_model,
            "topic_label_field": indexed_field
        },
        duplicate_detection={
            "tag_duplicates": detect_duplicates,
            "duplicate_cutoff": duplicate_cutoff
        }
    )
    
    return f"Successfully created index '{map_name}' for dataset '{dataset_identifier}'"

@mcp.tool()
def query_with_selections(
    dataset_identifier: str,
    filters: List[Dict[str, Any]],
    conjunctor: str = "ALL"
) -> List[Dict[str, Any]]:
    """
    Query a dataset with complex selections and filters

    dataset_identifier must be all lowercase no spaces (with a dash - for every space)
    
    Args:
        dataset_identifier: The project name from list_datasets
        filters: List of filter dictionaries. Each filter should have:
            - method: One of "search", "range", "category"
            - Other fields depending on method type
        conjunctor: How to combine filters - "ALL" (AND) or "ANY" (OR)
    """
    api_key = os.environ.get("NOMIC_API_KEY")
    if not api_key:
        return [{"error": "API key not found. Please set the NOMIC_API_KEY environment variable."}]
    
    atlas_dataset = AtlasDataset(dataset_identifier)
    payload = {
        "projection_id": atlas_dataset.maps[0].projection_id,
        "selection": {"method": "composition", "conjunctor": conjunctor, "filters": filters}
    }
    
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {api_key}'}
    response = requests.request("POST", "https://api-atlas.nomic.ai/v1/query", headers=headers, json=payload)
    return [{"error": f"Query request failed with status code {response.status_code}"}] if response.status_code != 200 else response.json()

@mcp.tool()
def vector_search_with_selections(
    dataset_identifier: str,
    query: Union[str, List[float]],
    k: int = 5,
    filters: Optional[List[Dict[str, Any]]] = None,
    return_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform vector search with optional filters

    dataset_identifier must be all lowercase no spaces (with a dash - for every space)
    
    Args:
        dataset_identifier: The project name from list_datasets
        query: Either a text string or embedding vector
        k: Number of results to return
        filters: Optional list of filter selections to apply before search
        return_fields: Optional list of fields to return in results
    """
    api_key = os.environ.get("NOMIC_API_KEY")
    if not api_key:
        return [{"error": "API key not found. Please set the NOMIC_API_KEY environment variable."}]
    
    atlas_dataset = AtlasDataset(dataset_identifier)
    payload = {"projection_id": atlas_dataset.maps[0].projection_id, "k": k, "query": query}
    
    if filters:
        payload["selection"] = {"method": "composition", "conjunctor": "ALL", "filters": filters}
    if return_fields:
        payload["fields"] = return_fields
    
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {api_key}'}
    response = requests.request("POST", "https://api-atlas.nomic.ai/v1/query/topk", headers=headers, json=payload)
    return [{"error": f"Search request failed with status code {response.status_code}"}] if response.status_code != 200 else response.json()

def make_search_filter(query: str, field: str, polarity: bool = True) -> Dict[str, Any]:
    """Create a search filter"""
    return {"method": "search", "query": query, "field": field, "polarity": polarity}

def make_range_filter(field: str, range_min: Union[int, float], range_max: Union[int, float]) -> Dict[str, Any]:
    """Create a range filter"""
    return {"method": "range", "field": field, "range": [range_min, range_max]}

def make_category_filter(field: str, values: List[str]) -> Dict[str, Any]:
    """Create a category filter"""
    return {"method": "category", "field": field, "values": values}

if __name__ == "__main__":
    mcp.run()