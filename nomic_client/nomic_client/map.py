import numpy as np
from typing import Any, Dict, List, Optional, Union, Literal

# from .utils import prepare_selection_payload


class AtlasMap:
    """Represents a map visualization of an Atlas dataset."""
    def __init__(self, client: 'NomicClient', dataset: 'AtlasDataset', map_id: str):
        """
        Initialize an Atlas map instance.

        Args:
            client: The NomicClient instance.
            dataset: The AtlasDataset this map belongs to.
            map_id: The identifier for this atlas_map.
        """
        self._client = client
        self._dataset = dataset
        self.id = map_id
        self._map_data = None

    def __repr__(self) -> str:
        """Return a string representation of the AtlasMap with key info."""
        lines = [
            f"┌ AtlasMap ─────────────────────────────────",
            f"│ Map ID: {self.id}",
            f"│ Dataset ID: {self._dataset.id}",
            f"│ Dataset Name: {self._dataset.name}",
            f"└────────────────────────────────────────────"
        ]
        return "\n".join(lines)

    def search(
        self,
        *,
        field: str,
        query: str,
        ignore_case: bool = True,
        regex: bool = False,
        word_boundary: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Performs a search query for the given text in the specified field.

        This is a convenience method that constructs a single 'search' selection
        and executes it using the `query` method.

        Args:
            field: The metadata field to search within.
            query: The text string to search for.
            ignore_case: Perform case-insensitive search (default: True).
            regex: Treat the query string as a regular expression (default: False).
            word_boundary: Match only whole words (default: False).

        Returns:
            List of dicts containing the datum id's of the
            matching data points.

        Example:
            # Enforces keyword arguments
            results = atlas_map.search(field='description', query='error')
        """
        search_selection = {
            "type": "search",
            "field": field,
            "query": query,
            "ignore_case": ignore_case,
            "regex": regex,
            "word_boundary": word_boundary,
        }
        return self.query(search_selection)

    def filter(
        self,
        *,
        field: str,
        value: Optional[Union[str, int, float, List[Union[str, int, float]]]] = None,
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs a query using a single category or range filter.

        This is a convenience method that constructs a single 'category' or 'range' selection
        based on the provided arguments and executes it using the `query` method.

        Args:
            field: The metadata field to filter.
            value: A single value or list of values for category filtering. Ignored if min_val/max_val are set.
                  Use this for exact matches (single value or list of allowed values).
            min_val: The inclusive minimum value for a range filter. Use None for no lower bound.
            max_val: The inclusive maximum value for a range filter. Use None for no upper bound.

        Returns:
            List of dicts containing the matching data points.

        Raises:
            ValueError: If inputs are ambiguous or invalid (e.g., no value/range specified,
                        empty list for value, or range selection validation fails).

        Example:
            # Filter for price less than or equal to 50 (Range)
            cheap_results = map.filter(field='price', max_val=50)

            # Filter for material being 'wood' or 'metal' (Category)
            wood_metal_results = map.filter(field='material', value=['wood', 'metal'])
        """
        is_range = min_val is not None or max_val is not None
        is_category = value is not None
        if is_range:
            selection = {
                "type": "range",
                "field": field,
                "min": min_val,
                "max": max_val
            }
        elif is_category:
            values_list = value if isinstance(value, list) else [value]
            selection = {
                "type": "category",
                "field": field,
                "values": values_list
            }
        else:
            raise ValueError("Filter requires 'value' (single or list) or 'min_val'/'max_val'.")
        return self.query(selection)

    # def vector_search(
    #     self,
    #     *,
    #     query: Union[str, List[float], np.ndarray],
    #     k: int = 10,
    #     selection: Optional[Dict[str, Any]] = None,
    #     return_fields: Optional[List[str]] = None,
    #     task_type: Optional[Literal["search_document", "search_query"]] = None,
    #     include_similarity: bool = True
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Search for the top k data points most semantically similar to a given query.

    #     Supports both text and vector queries. Optionally, apply selection before searching
    #     and specify which fields to return.

    #     Args:
    #         query: The search query. Can be a string (for text search) or a list/numpy array
    #                of floats (for vector search).
    #         k: The number of results to return.
    #         selection: An optional selection to apply before the vector search.
    #         return_fields: An optional list of metadata field names to include in the results.
    #                 If None, all user-uploaded fields are returned.
    #         task_type: Optional task type hint for embedding models (e.g., 'search_query',
    #                    'search_document').
    #         include_similarity: Whether to include the '_similarity' score in the results (default: True).

    #     Returns:
    #         A list of dicts, where each object represents a data point and includes
    #         the requested fields and potentially a '_similarity' score.

    #     Raises:
    #         ValueError: If the input parameters are invalid.
    #         TypeError: If the query type is incorrect or selection is not a valid dict.
    #         RuntimeError: If the API request fails.

    #     Examples:
    #         # Text search for 'comfortable footwear'
    #         results = atlas_map.vector_search(query='comfortable footwear', k=5)

    #         # Vector search using a numpy array, returning only 'title' and 'price'
    #         vector = np.random.rand(128)
    #         results = atlas_map.vector_search(query=vector, k=3, return_fields=['title', 'price'])
    #     """
    #     processed_query: Union[str, List[float]]
    #     if isinstance(query, np.ndarray):
    #         if query.ndim > 2:
    #             raise ValueError("query vector must be 1D or 2D numpy array")
    #         if query.ndim == 2 and query.shape[0] != 1:
    #             raise ValueError("2D query vector must have shape (1, N)")
    #         processed_query = query.flatten().tolist()
    #     elif isinstance(query, str) or (isinstance(query, list) and all(isinstance(item, (int, float)) for item in query)):
    #         processed_query = query
    #     else:
    #         raise TypeError("query must be a string, a list of numbers (int or float), or a 1D/2D numpy array")
    #     payload: Dict[str, Any] = {
    #         "projection_id": self.id,
    #         "k": k,
    #         "query": processed_query,
    #         "includeSimilarity": include_similarity
    #     }
    #     if return_fields is not None:
    #         payload["fields"] = return_fields
    #     if task_type is not None:
    #         payload["task_type"] = task_type
    #     if selection:
    #         selection_payload = prepare_selection_payload(selection)
    #         payload["selection"] = selection_payload
    #     response_data = self._client._post("query/topk", json_payload=payload)
    #     results = []
    #     for item in response_data["data"]:
    #         if not include_similarity:
    #             item.pop("_similarity", None)
    #         results.append(item)
    #     return results

    def query(
        self,
        selection: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Query data points using Atlas Selection DSL objects.

        Use helper methods like `self.search()`, `self.filter()`,
        `any_of()`, and `all_of()` to construct the filter criteria.

        Args:
            selection: A selection dict (e.g., search, category, range, or composition dict).

        Returns:
            List of dicts containing the datum id's of the
            matching data points. The 'datum' field in each result will be empty
            as this endpoint only returns IDs.

        Example:
            # Find chairs that are either under $100 OR made of wood
            selection = {
                "type": "all_of",
                "selections": [
                    {"type": "search", "field": "title", "query": "chair"},
                    {"type": "any_of", "selections": [
                        {"type": "range", "field": "price", "max": 100},
                        {"type": "category", "field": "material", "values": ["wood"]}
                    ]}
                ]
            }
            results = atlas_map.query(selection)

        Raises:
            TypeError: If the `selection` argument is not a valid dict.
            ValueError: If the provided selection is invalid or the API request fails.
            RuntimeError: If the API request fails or returns an unexpected response.
        """
        selection_dsl_json = prepare_selection_payload(selection)
        payload = {
            "projection_id": self.id,
            "selection": selection_dsl_json
        }
        response_data = self._client._post("query", json_payload=payload)
        return response_data["data"]
