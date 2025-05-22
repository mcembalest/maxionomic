from nomic_client import NomicClient

client = NomicClient()
print(client.user)

atlas_dataset = client.load_dataset("airline-reviews-data")
print(atlas_dataset)
















# import numpy as np
# import pandas as pd
# import pyarrow as pa
# from datetime import datetime, timedelta
# from utils import normalize, format_iso8601



# # Create dataset
# now = datetime.now()
# num_rows = 100
# text1_values = [f"Sample text {i}" for i in range(num_rows)]
# text2_values = [f"Another text {i}" for i in range(num_rows)]
# score_values = np.random.rand(num_rows).tolist()
# created_at_values = [format_iso8601(now - timedelta(days=i)) for i in range(num_rows)]
# embeddings_values = []
# for _ in range(num_rows):
#     raw_embedding = normalize(2 * np.random.rand(768) - 1)
#     embeddings_values.append(raw_embedding.tolist())

# # Create PyArrow arrays
# text1_array = pa.array(text1_values, type=pa.string())
# text2_array = pa.array(text2_values, type=pa.string())
# score_array = pa.array(score_values, type=pa.float64())
# created_at_array = pa.array(created_at_values, type=pa.string())
# embeddings_array = pa.array(embeddings_values, type=pa.list_(pa.float64()))

# # Create PyArrow table
# table = pa.table([
#     text1_array, 
#     text2_array, 
#     score_array,
#     created_at_array,
#     embeddings_array
# ], names=['text1', 'text2', 'score', 'created_at', 'embeddings'])

# print(table.schema)

# # Create dataset and add data
# new_dataset = client.create_dataset("demo-dataset")
# print(new_dataset)

# result = new_dataset.add_data(table)
# print(result)

# # Create a map from the text field
# atlas_map = new_dataset.create_map(field_to_embed="text1")
# print("\nCreated Map:")
# print(atlas_map)


