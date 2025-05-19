import os
import anthropic
from nomic import AtlasDataset
import pandas as pd 
import requests

NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL_NAME = "claude-3-7-sonnet-20250219" 

def fetch_atlas_dataframes():
    """
    Fetches ground truth data, Nomic Atlas topic model data, and the AtlasMap object.
    Returns three items: data_df, topics_df, and atlas_map.
    """
    print("Fetching data from Nomic Atlas...")
    atlas_dataset = AtlasDataset('ai-policy-recommendations')
    atlas_map = atlas_dataset.maps[0]
    data_df = atlas_map.data.df[['Details', 'Topic']].copy()
    print("Successfully fetched ground truth data (data.df).")
    topics_df = atlas_map.topics.df.copy()
    print("Successfully fetched Nomic Atlas topics (topics.df).")
    return data_df, topics_df, atlas_map
    
def display_topic_comparison(data_df, topics_df, num_samples=3):
    """
    Constructs and returns a DataFrame comparing ground truth topics, Nomic Atlas topics,
    and a placeholder for the proposed GeoJSON + Claude Sonnet topics.
    """
    cols_to_join = ['topic_depth_1', 'topic_depth_2']
    topics_df_subset = topics_df[[col for col in cols_to_join if col in topics_df.columns]]
    merged_df = data_df.join(topics_df_subset, how='left') 
    sample_df = merged_df.sample(min(num_samples, len(merged_df)))
    comparison_data = []
    for index, row in sample_df.iterrows():
        comparison_data.append({
            'Datum ID': index,
            'Details': str(row.get('Details', "N/A"))[:40] + "...",
            'NomicAtlasTopicDepth1': row.get('topic_depth_1', "N/A"),
            'NomicAtlasTopicDepth2': row.get('topic_depth_2', "N/A"),
            'ProposedClaudeTopic': "TODO",
            'GroundTruthTopic': row.get('Topic', "N/A"),
        })
    comparison_df = pd.DataFrame(comparison_data)
    column_order = ['ItemIndex', 'Details', 'GroundTruthTopic', 'NomicAtlasTopicDepth1', 'NomicAtlasTopicDepth2', 'ProposedClaudeTopic']
    return comparison_df[[col for col in column_order if col in comparison_df.columns]]

def get_claude_topic_label(client, text_content, model_name=CLAUDE_MODEL_NAME):
    """
    Gets a topic label for the given text_content using Claude.
    """
    system_prompt = "You are an expert in topic modeling. Given the following text, identify and return a concise topic label (2-5 words). Respond only with the topic label itself, without any preamble or explanation."
    
    response = client.messages.create(
        model=model_name,
        max_tokens=20,
        temperature=0.5,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please provide a topic label for the following text:\n\n{text_content}"
                    }
                ]
            }
        ]
    )
    return response.content[0].text

def label_items_with_claude_and_display(df_ground_truth, client, num_items_to_label=3):
    """
    Labels a sample of items from the DataFrame using Claude and displays results.
    """
    print(f"\n--- Claude Sonnet Topic Labeling (Sample of {min(num_items_to_label, len(df_ground_truth))} items) ---")    
    sample_df = df_ground_truth.sample(min(num_items_to_label, len(df_ground_truth)))
    for index, row in sample_df.iterrows():
        details_text = row['Details']
        ground_truth_topic = row['Topic']
        print(f"\nProcessing item (Index: {index}):")
        print(f"  Details: {details_text[:50]}...")
        print(f"  Ground Truth Topic: {ground_truth_topic}")
        claude_label = get_claude_topic_label(client, details_text)
        print(f"  Claude Sonnet Generated Topic: {claude_label}")

def get_topic_datum_counts(atlas_map, depths_to_fetch=[1, 2, 3]):
    """
    Fetches topic details for specified depths and returns a dictionary
    mapping (depth, topic_short_description) to datum_count.
    """
    topic_counts = {}
    for depth_level in depths_to_fetch:
        topics_at_depth = atlas_map.topics.group_by_topic(depth_level)
        for topic_info in topics_at_depth:
            short_desc = topic_info.get('topic_short_description')
            datum_ids = topic_info.get('datum_ids')
            if short_desc and datum_ids is not None:
                topic_counts[(depth_level, short_desc)] = len(datum_ids)
            else:
                print(f"Warning: Missing 'topic_short_description' or 'datum_ids' for a topic at depth {depth_level}.")
    return topic_counts

def display_topic_hierarchy(atlas_map):
    """
    Displays the parent-child relationships between depth 1 and depth 2 topics.
    """
    print("\n--- Nomic Atlas Topic Model Hierarchy (Depth 1 to Depth 2) ---")
    hierarchy = atlas_map.topics.hierarchy
    for topic_key, sub_topics in hierarchy.items():
        # topic_key is a tuple like ('Topic Name', depth)
        topic_name, topic_depth = topic_key
        if topic_depth == 1: # We are interested in Depth 1 topics as parents
            print(f"\nParent Topic (Depth 1): {topic_name}")
            if sub_topics:
                print("  Child Topics (Depth 2):")
                # sub_topics can be a list of strings (child topic names) or a list of tuples/dicts
                # Based on the documentation, it seems to be a list of child topic names for the next depth.
                for sub_topic_name in sub_topics:
                    # The hierarchy structure might directly give names, or names within tuples.
                    # Adjusting based on typical output: if sub_topic is ('Child Name', 2), extract 'Child Name'
                    if isinstance(sub_topic_name, tuple):
                        print(f"    - {sub_topic_name[0]}")
                    else:
                        print(f"    - {sub_topic_name}")
            else:
                print("  No direct child topics (Depth 2) listed in hierarchy for this topic.")

if __name__ == "__main__":
    print("Starting script...")
    topic_datum_counts = {} 
    data_df, topics_df, atlas_map = fetch_atlas_dataframes()
    if data_df is not None and topics_df is not None:
        comparison_result_df = display_topic_comparison(data_df, topics_df, num_samples=5)
        if comparison_result_df is not None:
            print("\n--- Topic Comparison DataFrame ---")
            print(comparison_result_df.to_string())
    topic_datum_counts = get_topic_datum_counts(atlas_map, depths_to_fetch=[1, 2])
    

    # Get Atlas topic model clustering geojson 
    # (without labels, trying to use Claude Sonnet to relabel so we can benchmark these topics against ground truth)
    project_id = "f2bba6e5-e3df-46ef-ad79-7362bd3c48f5"
    projection_id = "1abba3fa-b89d-40b3-a0e2-0cf52b2a7f13"
    api_url = f"https://api-atlas.nomic.ai/v1/project/{project_id}/index/projection/{projection_id}"
    response = requests.get(
        api_url,
        headers={"Authorization": f"Bearer {NOMIC_API_KEY}"}
    )
    topic_models_geojson = response.json()
    num_feature_collections = len(topic_models_geojson['topic_models'])
    for i, feature_collection in enumerate(topic_models_geojson['topic_models']):
        features = feature_collection['features']
        depth_counts = {}
        overall_unique_names_by_topic_field_level = {} 
        unique_names_by_feature_depth = {}
        short_desc_lengths = []
        desc_lengths = []
        for feature in features:
            properties = feature['properties']
            
            actual_feature_depth = properties.get('depth')
            if actual_feature_depth is not None:
                depth_counts[actual_feature_depth] = depth_counts.get(actual_feature_depth, 0) + 1
                
                topic_key_for_current_depth = f"topic_depth_{actual_feature_depth}"
                if topic_key_for_current_depth in properties:
                    if actual_feature_depth not in unique_names_by_feature_depth:
                        unique_names_by_feature_depth[actual_feature_depth] = set()
                    unique_names_by_feature_depth[actual_feature_depth].add(properties[topic_key_for_current_depth])

            for key, value in properties.items():
                if key.startswith('topic_depth_'):
                    level = int(key.split('_')[-1])
                    if level not in overall_unique_names_by_topic_field_level:
                        overall_unique_names_by_topic_field_level[level] = set()
                    overall_unique_names_by_topic_field_level[level].add(value)
            
            if 'topic_short_description' in properties:
                short_desc_lengths.append(len(properties['topic_short_description'] or ''))
            if 'topic_description' in properties:
                desc_lengths.append(len(properties['topic_description'] or ''))
        print('---------------------------------')
        print("Nomic Atlas Topic Model summary statistics")
        for d, count in sorted(depth_counts.items()):
            print(f"    Features with depth {d}: {count}")    
        avg_short_desc_len = sum(short_desc_lengths) / len(short_desc_lengths) if short_desc_lengths else 0
        print(f"    Average 'topic_short_description' length: {avg_short_desc_len:.2f} characters (for {len(short_desc_lengths)} descriptions)")
        print('---------------------------------')
        display_topic_hierarchy(atlas_map)
        print("===================Details for Depth 1 Topics:")
        depth_1_topic_details_found = False
        for feature_idx, feature in enumerate(features):
            if 'properties' in feature and isinstance(feature['properties'], dict):
                properties = feature['properties']
                if properties.get('depth') == 1:
                    depth_1_topic_details_found = True
                    short_desc = properties.get('topic_short_description', 'N/A')
                    full_desc = properties.get('topic_description', 'N/A')
                    datum_count = topic_datum_counts.get((1, short_desc), "N/A")
                    print(f"    Topic {feature_idx + 1} (Depth 1):")
                    print(f"      Label: {short_desc} (Datums: {datum_count})")
                    print(f"      Keywords: {full_desc}")
        
        if not depth_1_topic_details_found:
            print("    No topics found at depth 1.")
        
        print("===================Details for Depth 2 Topics:")
        depth_2_features = []
        for feature in features:
            if 'properties' in feature and isinstance(feature['properties'], dict):
                properties = feature['properties']
                if properties.get('depth') == 2:
                    depth_2_features.append(feature)

        num_depth_2_features = len(depth_2_features)        
        for feature_idx, feature in enumerate(depth_2_features):
            properties = feature['properties'] 
            short_desc = properties.get('topic_short_description', 'N/A')
            full_desc = properties.get('topic_description', 'N/A')
            actual_feature_depth = properties.get('depth') 
            datum_count = topic_datum_counts.get((actual_feature_depth, short_desc), "N/A")
            print(f"    Topic {feature_idx + 1} (Depth {actual_feature_depth}):") 
            print(f"      Label: {short_desc} (Datums: {datum_count})")
            print(f"      Keywords: {full_desc}")


    # anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    # label_items_with_claude_and_display(data_df, anthropic_client)