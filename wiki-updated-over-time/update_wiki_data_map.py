import os
import pandas as pd
from nomic import AtlasDataset

def load_downloaded_articles():
    """Load article content from downloaded_articles directory"""
    articles = []
    for filename in sorted(os.listdir("downloaded_articles")):
        if filename.endswith(".txt"):
            file_path = os.path.join("downloaded_articles", filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                page_id = filename.split('_')[0]
                articles.append({
                    'id': page_id,
                    'text': content
                })
    articles.sort(key=lambda x: x['id'])
    return articles

dataset = AtlasDataset(
    "wiki-updated-over-time",
    unique_id_field="id",
)

existing_ids = dataset.maps[0].data.df.id.values
print(existing_ids)

downloaded_data = load_downloaded_articles()
new_data = [x for x in downloaded_data if x['id'] not in existing_ids]
print(" "*20, "Adding", len(new_data), "new data points to map")
dataset.add_data(new_data)
dataset.create_index(indexed_field='text')


