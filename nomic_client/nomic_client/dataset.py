import base64
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import pyarrow as pa

from .map import AtlasMap
from .utils import (_convert_to_arrow, 
                    _validate_atlas_schema,
                    _prepare_special_columns_schema,
                    )


class AtlasDataset:
    """
    Represents a dataset in Nomic Atlas. 
    Usually created via `nomic.NomicClient.load_dataset()` or `nomic.NomicClient.create_dataset()`.
    """
    def __init__(
        self,
        client: 'NomicClient',
        id: str,
        name: str,
        is_public: bool,
        created_at: str,
        organization_name: str,
        organization_slug: str,
        description: Optional[str] = "",
        schema: Optional[Any] = None,
        total_datums: Optional[int] = None,
    ):
        self.client = client
        self.id = id
        self.name = name
        self.description = description
        self.is_public = is_public
        self.created_at = created_at
        self.organization_name = organization_name
        self.organization_slug = organization_slug
        self._schema = schema
        self.total_datums = total_datums

    def __repr__(self) -> str:
        """Return a string representation of the AtlasDataset with key info."""
        description_text = self.description if self.description is not None else ""
        lines = [
            f"┌ AtlasDataset ──────────────────────────────",
            f"│ ID: {self.id}",
            f"│ Name: {self.name}",
            f"│ Description: {description_text[:50] + '...' if len(description_text) > 50 else description_text}",
            f"│ Public: {'✓' if self.is_public else '✗'}",
            f"│ Created: {self.created_at}",
            f"│ Organization: {self.organization_name}",
            f"└────────────────────────────────────────────"
        ]
        return "\n".join(lines)

    @property
    def schema(self) -> Optional[pa.Schema]:
        """Dataset schema."""
        if self._schema is None:
            try:
                schema_b64 = self.client._get_dataset(self.id).get("schema")
                if schema_b64 and isinstance(schema_b64, str):
                    self._schema = schema_b64
                else:
                    self._schema = None
            except Exception as e:
                print(f"Warning: Failed to fetch schema for dataset {self.id}: {e}")
                return None
        if isinstance(self._schema, str):
            try:
                schema_bytes = base64.b64decode(self._schema)
                schema_buffer = pa.BufferReader(schema_bytes)
                return pa.ipc.read_schema(schema_buffer)
            except Exception as e:
                print(f"Warning: Failed to decode schema from base64: {e}")
                return None
        elif isinstance(self._schema, pa.Schema):
            return self._schema
        return None

    @classmethod
    def _create(cls, client: 'NomicClient', dataset_info: Dict[str, Any]) -> 'AtlasDataset':
        """(Private) Factory method to create an AtlasDataset.
        """
        
        return cls(
            client=client,
            id=dataset_info.get("id"),
            name=dataset_info.get("project_name"),
            description=dataset_info.get("description"),
            is_public=dataset_info.get("is_public"),
            created_at=dataset_info.get("created_timestamp"),
            organization_name=dataset_info.get("organization_name"),
            organization_slug=dataset_info.get("organization_slug"),
            schema=dataset_info.get("schema"),
        )

    def add_data(
        self,
        data: Union[Dict[str, List], List[Dict[str, Any]], 'pd.DataFrame', pa.Table],
    ) -> Dict[str, Any]:
        """
        Add data to the dataset. Accepts various common Python data structures
        and automatically handles image and embedding columns.

        Image data can be provided via columns named 'image_path' (containing file paths)
        or 'image_pil' (containing PIL Image objects). These columns will be
        processed, uploaded, and replaced by a '_blob_hash' column.
        Pre-computed embeddings can be provided via a column named 'embeddings'
        containing vectors (e.g., numpy arrays, lists of floats, or Arrow list arrays).
        These will be validated and formatted appropriately.

        Args:
            data: The data to add. Can be:
                - A dictionary of lists (column-oriented: {"col1": [1,2,3], "col2": ["a","b","c"]})
                - A list of dictionaries (row-oriented: [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}])
                - A pandas DataFrame
                - A PyArrow Table

        Returns:
            Dictionary containing the API response with information about the upload.

        Example:
            ```python
            # Add metadata with image paths
            dataset.add_data([
                {"id": "img1", "path": "/path/to/image1.jpg", "label": "cat"},
                {"id": "img2", "path": "/path/to/image2.png", "label": "dog"}
            ])

            # Add metadata with pre-computed embeddings
            import numpy as np
            dataset.add_data({
                "id": ["vec1", "vec2"],
                "embeddings": [np.random.rand(10), np.random.rand(10)],
                "text": ["text 1", "text 2"]
            })
            ```
        """
        table = _convert_to_arrow(data)
        _validate_atlas_schema(table)
        prepared_table, image_col_names_found = _prepare_special_columns_schema(table)
        current_schema = self.schema
        if current_schema is None:
            self._schema = prepared_table.schema
            current_schema = prepared_table.schema
        if not prepared_table.schema.equals(current_schema, check_metadata=False):
            print("\nSchema Mismatch Detected!")
            print("Existing Schema:")
            print(current_schema)
            print("\nNew Data Schema (after preparing special columns):")
            print(prepared_table.schema)
            raise ValueError("Schema mismatch. All data added must conform to the established dataset schema.")
        final_table = prepared_table
        initial_size = self.client._get_dataset_size(self.id)
        if image_col_names_found:
            index_to_hash = self.client._process_and_upload_images_for_dataset(
                dataset_id=self.id,
                table=table, 
                image_col_names=image_col_names_found,
            )
            if index_to_hash:
                blob_hashes = [index_to_hash.get(i) for i in range(len(final_table))]
                hash_array = pa.array(blob_hashes, type=pa.string())
                blob_hash_field_index = final_table.schema.get_field_index('_blob_hash')
                final_table = final_table.set_column(blob_hash_field_index, '_blob_hash', hash_array)
        meta = dict(final_table.schema.metadata or {})
        meta[b"project_id"] = self.id.encode('utf-8')
        final_table = final_table.replace_schema_metadata(meta)
        sink = pa.BufferOutputStream()
        write_options = pa.ipc.IpcWriteOptions(compression='zstd')
        with pa.ipc.new_file(sink, final_table.schema, options=write_options) as writer:
            writer.write_table(final_table)
        body = sink.getvalue().to_pybytes()
        upload_size_mb = len(body) / (1024 * 1024)
        print(f"Uploading {len(final_table)} prepared rows ({upload_size_mb:.2f} MB)...")
        response_data = self.client._add_arrow_data(self.id, body)
        status_msg = response_data.get('message')
        print(f"Upload status: {status_msg or 'OK'}")
        final_size = self.client._get_dataset_size(self.id)
        num_rows_added = None
        if initial_size is not None and final_size is not None:
            num_rows_added = final_size - initial_size
        print(f"Added {num_rows_added} rows to dataset {self.id}.")

    def create_map(self, field_to_embed: str) -> AtlasMap:
        """Build a new Atlas Map from the specified column.
        
        Args:
            field_to_embed: The column containing text data to use for creating the map.
        
        Returns:
            AtlasMap: The newly created map
        """
        colorable_fields = []
        current_schema = self.schema
        for field in current_schema: # For now, assume non-id, non-embedding, non-blob fields are colorable
            if field.name not in [field_to_embed, 'embeddings', '_blob_hash'] and not field.name.startswith('_'):
                colorable_fields.append(field.name)
        payload = {
            "project_id": self.id,
            "index_name": None,
            "indexed_field": field_to_embed,
            "atomizer_strategies": ["document", "charchunk"],
            "model": "nomic-embed-text-v1.5",
            "colorable_fields": colorable_fields,
            "model_hyperparameters": json.dumps({ 
                "dataset_buffer_size": 1000,
                "batch_size": 20,
                "polymerize_by": "charchunk",
                "norm": "both",
            }),
            "nearest_neighbor_index": "HNSWIndex",
            "nearest_neighbor_index_hyperparameters": json.dumps({"space": "l2", "ef_construction": 100, "M": 16}),
            "projection": "NomicProject",
            "projection_hyperparameters": json.dumps({
                "n_neighbors": 15,
                "n_epochs": 50,
                "spread": 1.0,
            }),
            "topic_model_hyperparameters": json.dumps({ 
                "build_topic_model": True,
                "community_description_target_field": field_to_embed,
                "cluster_method": "fastidious",
                "enforce_topic_hierarchy": False
            }),
            "duplicate_detection_hyperparameters": json.dumps({ 
                "tag_duplicates": True,
                "duplicate_cutoff": 0.75
            })
        }
        data = self.client._create_index(payload)
        job_id = data.get("job_id")
        job_data = self.client._get_index_job_status(job_id)
        projection_id = job_data.get("projection_id")
        return AtlasMap(client=self.client, dataset=self, map_id=projection_id)
    
    def load_map(self, map_id: Optional[str] = None) -> AtlasMap:
        """
        Get an AtlasMap object by its map ID.
        If map_id is not provided, loads the most recently created map based on creation timestamp.
        """
        if map_id:
            print(f"Loading map with specified ID: {map_id}")
            return AtlasMap(client=self.client, dataset=self, map_id=map_id)
        print("Loading the most chronologically recent map...")
        atlas_indices = self.client._get_dataset(self.id).get("atlas_indices", [])
        latest_projection = None
        latest_ts = None
        if not atlas_indices:
            raise ValueError(f"No maps found for dataset {self.id}.")
        for index in atlas_indices:
            projections = index.get("projections", [])
            for projection in projections:
                timestamp_string = projection.get("created_timestamp")
                proj_id = projection.get("id")
                if not proj_id:
                    continue
                if timestamp_string:
                    if timestamp_string.endswith('Z'):
                        timestamp_string = timestamp_string[:-1] + '+00:00'
                    current_ts = datetime.fromisoformat(timestamp_string)
                    if latest_ts is None or current_ts > latest_ts:
                        latest_ts = current_ts
                        latest_projection = projection
        if latest_projection:
            map_id = latest_projection.get("id")
            if map_id:
                print(f"Found most recent map: ID {map_id}, Created: {latest_ts}")
                return AtlasMap(client=self.client, dataset=self, map_id=map_id)
            else:
                raise ValueError("Identified latest map, but it unexpectedly lacks an 'id'.")
        else:
            raise ValueError(f"Could not find any valid maps with creation timestamps for dataset {self.id}.")

    def delete(self) -> None:
        """Delete the dataset and all its maps from Nomic Atlas."""
        try:
            self.client._delete_project(self.id)
            print(f"Dataset {self.id} deleted successfully.")
        except RuntimeError as e:
            raise
