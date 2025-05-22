from typing import Any, Dict, List, Literal, Union, Optional, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np

def _convert_to_arrow(data: Union[Dict[str, List], List[Dict[str, Any]], pd.DataFrame, pa.Table]) -> pa.Table:
    """Converts various Python data structures to a PyArrow Table."""
    if isinstance(data, pa.Table):
        return data
    elif isinstance(data, pd.DataFrame):
        schema_overrides = {name: pa.string() for name, dtype in data.dtypes.items() if dtype == 'object'}
        return pa.Table.from_pandas(data, schema=pa.schema(schema_overrides) if schema_overrides else None, preserve_index=False)
    elif isinstance(data, dict):
        return pa.Table.from_pydict(data)
    elif isinstance(data, list) and data:
        return pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False)
    else:
        raise ValueError(
            "Unsupported data format. Use PyArrow Table, pandas DataFrame, "
            "dictionary of lists, or non-empty list of dictionaries."
        )

def _validate_atlas_schema(table: pa.Table) -> None:
    """
    Validates the schema of a PyArrow Table against Nomic Atlas rules.

    Args:
        table: The PyArrow Table to validate.

    Raises:
        ValueError: If the schema violates Atlas rules (e.g., duplicate field names,
                    reserved field names).
    """
    field_names_lower = [field.name.lower() for field in table.schema]
    if len(field_names_lower) != len(set(field_names_lower)):
        seen = set()
        duplicates = []
        original_names = [field.name for field in table.schema]
        lower_to_original = {name.lower(): name for name in original_names}
        for name_lower in field_names_lower:
            if name_lower in seen:
                duplicates.append(lower_to_original[name_lower])
            seen.add(name_lower)
        raise ValueError(f"Found duplicate field names (case-insensitive): {list(set(duplicates))}. Field names must be unique.")
    reserved_prefixes = ['_']
    allowed_underscore_fields = {'_blob_hash'}
    for field in table.schema:
        if any(field.name.startswith(prefix) for prefix in reserved_prefixes) and \
           field.name not in allowed_underscore_fields:
            raise ValueError(f"Field names starting with underscore (_) are reserved for Atlas internal use. Found: {field.name}")


def _validate_and_prepare_embeddings_column(table: pa.Table) -> pa.Table:
    """Validates and formats the 'embeddings' column if present, returning a new table."""
    if "embeddings" not in table.column_names:
        return table

    print("Validating and formatting 'embeddings' column...")
    col_name = "embeddings"
    embeddings_col = table[col_name]

    # Attempt conversion if not already list-like
    if not pa.types.is_list_like(embeddings_col.type):
        try:
            embeddings_np = np.array(embeddings_col.to_pylist(), dtype=object)
            # Filter out Nones before processing
            valid_embeddings = [emb for emb in embeddings_np if emb is not None]
            if not valid_embeddings:
                # Handle case where all embeddings are None - keep as null array or specific type?
                # For now, let's just return table as is if only nulls present before conversion
                 print("Warning: 'embeddings' column contains only null values or non-convertible types.")
                 return table # Or raise error?

            # Process non-None embeddings
            processed_embeddings = []
            for emb in embeddings_np:
                if emb is None:
                    processed_embeddings.append(None)
                else:
                    # Convert each element robustly
                    try:
                        processed_embeddings.append(np.asarray(emb, dtype=np.float32))
                    except ValueError as e:
                         raise TypeError(f"Could not convert element in 'embeddings' to float32 array: {emb}. Error: {e}") from e

            # Check dimensions consistency only on non-None processed embeddings
            non_none_processed = [e for e in processed_embeddings if e is not None]
            dims = {e.shape[0] for e in non_none_processed if hasattr(e, 'shape') and len(e.shape) > 0}

            if len(dims) > 1:
                raise ValueError(f"Embeddings have inconsistent dimensions: {dims}")
            dim = dims.pop() if dims else 0 # Handle case of only Nones or empty arrays

            # Determine target type (float16 default, float32 for 2D)
            # Use float32 as default for better precision compatibility
            target_value_type = pa.float32() # Defaulting to float32
            # target_value_type = pa.float16() if dim != 2 else pa.float32()

            # Create Arrow array - handle potential all-None case after processing
            try:
                if dim > 0:
                    # Build list array first, then cast to fixed size if needed
                    list_array = pa.array(processed_embeddings, type=pa.list_(target_value_type))
                    embeddings_col = list_array.cast(pa.fixed_size_list(target_value_type, dim))
                else: # Case where dim is 0 (e.g., all Nones, or empty lists)
                     embeddings_col = pa.nulls(len(table), type=pa.list_(target_value_type)) # Create a null array of list type
                print(f"Converted embeddings column from source type to: {embeddings_col.type}")
            except (pa.ArrowException, TypeError, ValueError) as e:
                 raise TypeError(f"Failed to create Arrow Array for embeddings. Dimension: {dim}, Type: {target_value_type}. Error: {e}") from e

        except Exception as e:
            raise TypeError(f"Column 'embeddings' has unsupported type {embeddings_col.type} and could not be auto-converted. Provide embeddings as lists/arrays of numbers or a PyArrow ListArray/FixedSizeListArray. Original error: {e}") from e

    # Ensure value type is floating
    if not pa.types.is_list_like(embeddings_col.type) or not pa.types.is_floating(embeddings_col.type.value_type):
         try:
             # Attempt to cast inner values if needed (e.g., int list to float list)
             target_type = pa.float32() # Consistently use float32
             # if hasattr(embeddings_col.type, 'list_size') and embeddings_col.type.list_size == 2:
             #     target_type = pa.float32()
             print(f"Casting embedding values from {embeddings_col.type.value_type} to {target_type}...")
             list_type = pa.list_(target_type)
             embeddings_col = embeddings_col.cast(list_type)
             # Re-apply fixed size list if applicable
             if hasattr(embeddings_col.type, 'list_size'): # Check if original was fixed
                 dim = embeddings_col.type.list_size
                 embeddings_col = embeddings_col.cast(pa.fixed_size_list(target_type, dim))
             elif pa.types.is_fixed_size_list(embeddings_col.type): # Check if cast resulted in fixed
                  dim = embeddings_col.type.list_size
                  embeddings_col = embeddings_col.cast(pa.fixed_size_list(target_type, dim))

         except (pa.ArrowInvalid, TypeError) as e:
             raise TypeError(f"Embeddings column inner type must be numeric (float, int, etc.), found {embeddings_col.type.value_type}. Could not auto-cast to float. Error: {e}") from e

    # Check for NaN/Inf in the final floating point values
    if pa.types.is_floating(embeddings_col.type.value_type):
        # Need to handle potential nulls in the list array itself before accessing .values
        flat_embeddings = embeddings_col.values if hasattr(embeddings_col, 'values') else None
        if flat_embeddings is not None and len(flat_embeddings) > 0:
            # Check only non-null values within the flattened array
            valid_mask = pc.invert(pc.is_null(flat_embeddings))
            valid_values = pc.filter(flat_embeddings, valid_mask)
            if len(valid_values) > 0:
                has_nan = pc.any(pc.is_nan(valid_values)).as_py()
                has_inf = pc.any(pc.is_inf(valid_values)).as_py()
                if has_nan or has_inf:
                    raise ValueError("Embeddings column contains NaN or Inf values. Please clean the data before uploading.")

    # Replace the column in the table
    col_index = table.schema.get_field_index(col_name)
    # Ensure the field allows nulls, as processing might introduce them
    new_field = pa.field(col_name, embeddings_col.type, nullable=True)
    table = table.set_column(col_index, new_field, embeddings_col)
    print("Embeddings column validated and formatted successfully.")
    return table

def _prepare_special_columns_schema(table: pa.Table) -> Tuple[pa.Table, Optional[List[str]]]:
    """
    Prepares the table schema for special columns (embeddings, images) without uploading images.

    Validates/prepares the 'embeddings' column.
    Identifies image columns ('image_path', 'image_pil'), removes them, 
    and adds a placeholder '_blob_hash' column if image columns are present.

    Args:
        table: The input PyArrow Table.

    Returns:
        A tuple containing:
        - The processed PyArrow Table with schema modifications.
        - A list of original image column names found, or None if none were found.
    
    Raises:
        ValueError: If embedding processing fails.
    """
    image_col_names = [name for name in table.column_names if name == 'image_path' or name == 'image_pil']
    processed_table = table
    original_image_col_names = None

    if image_col_names:
        print(f"Detected image columns for schema preparation: {image_col_names}")
        original_image_col_names = image_col_names # Store the names
        print(f"Removing original image columns from schema: {image_col_names}")
        processed_table = processed_table.drop_columns(image_col_names)
        print("Adding placeholder _blob_hash column to schema...")
        # Add a column of nulls with string type as a placeholder
        placeholder_hashes = pa.nulls(len(processed_table), pa.string())
        processed_table = processed_table.append_column('_blob_hash', placeholder_hashes)
        print("Schema prepared for images.")

    # Embedding Handling (can modify the table further)
    if 'embeddings' in processed_table.column_names:
        print("Processing embeddings column schema...")
        try:
            processed_table = _validate_and_prepare_embeddings_column(processed_table)
        except Exception as e:
            print(f"Error processing embeddings: {e}")
            raise # Re-raise the exception after logging
    
    return processed_table, original_image_col_names

# def _combine_selections(
#     conjunctor: Literal['ANY', 'ALL'],
#     *filters: SelectionDSL
# ) -> CompositionSelection:
#     """Internal helper to create a composition selection."""
#     if not filters:
#         raise ValueError("At least one filter must be provided for composition.")
#     valid_filters = []
#     for f in filters:
#         if isinstance(f, (SearchSelection, RangeSelection, CategorySelection, CompositionSelection)):
#             valid_filters.append(f)
#         else:
#             raise TypeError(f"Expected SelectionDSL objects, but got: {type(f)}")
#     return CompositionSelection(
#         conjunctor=conjunctor,
#         filters=list(valid_filters),
#         polarity=True,
#     )

# def any_of(*filters: SelectionDSL) -> CompositionSelection:
#     """
#     Combines multiple selections using a logical OR (ANY).

#     Args:
#         *filters: Two or more SelectionDSL objects (e.g., from search(), filter()).

#     Returns:
#         A CompositionSelection object representing the combined selections.
#     """
#     return _combine_selections('ANY', *filters)

# def all_of(*filters: SelectionDSL) -> CompositionSelection:
#     """
#     Combines multiple selections using a logical AND (ALL).

#     Args:
#         *filters: Two or more SelectionDSL objects (e.g., from search(), filter()).

#     Returns:
#         A CompositionSelection object representing the combined selections.
#     """
#     return _combine_selections('ALL', *filters)

# def _serialize_selection_dsl(selection: SelectionDSL) -> Dict[str, Any]:
#     """Converts a SelectionDSL dataclass object into the JSON dictionary format expected by the Nomic API."""
#     if isinstance(selection, SearchSelection):
#         return {
#             "method": selection.method,
#             "field": selection.field,
#             "query": selection.query,
#             "ignoreCase": selection.ignore_case,
#             "regex": selection.regex,
#             "wordBoundary": selection.word_boundary,
#         }
#     elif isinstance(selection, RangeSelection):
#         return {
#             "method": selection.method,
#             "field": selection.field,
#             "range": [selection.min_val, selection.max_val],
#         }
#     elif isinstance(selection, CategorySelection):
#         return {
#             "method": selection.method,
#             "field": selection.field,
#             "values": selection.values,
#         }
#     elif isinstance(selection, CompositionSelection):
#         return {
#             "method": selection.method,
#             "conjunctor": selection.conjunctor,
#             "filters": [_serialize_selection_dsl(f) for f in selection.filters], # Recursively serialize nested filters
#             "polarity": selection.polarity,
#         }
#     else:
#         raise TypeError(f"Unsupported selection type: {type(selection)}")

# def prepare_selection_payload(selection: Optional[SelectionDSL]) -> Optional[Dict[str, Any]]:
#     """Validates and prepares the selection DSL dictionary for API payload.

#     Ensures the top-level structure is always a composition.
#     """
#     if selection is None:
#         return None
#     if not isinstance(selection, (SearchSelection, RangeSelection, CategorySelection, CompositionSelection)):
#         raise TypeError("`selection` must be a valid SelectionDSL object (e.g., created by search(), filter(), any_of(), all_of()).")
#     selection_dsl = _serialize_selection_dsl(selection)
#     if selection_dsl.get("method") != "composition":
#          selection_dsl = {
#              "method": "composition",
#              "conjunctor": "ALL",
#              "filters": [selection_dsl],
#          }
#     return selection_dsl
