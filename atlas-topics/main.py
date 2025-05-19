import os
import anthropic
from nomic import AtlasDataset
import pandas as pd 
import requests

NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL_NAME = "claude-3-7-sonnet-20250219" 

def get_topic_datum_counts(atlas_map, depths_to_fetch=[1, 2, 3]):
    """
    Fetches topic info for specified depths and returns a dictionary
    mapping (depth, topic_short_description) to datum_count.
    """
    topic_counts = {}
    for depth_level in depths_to_fetch:
        topics_at_depth = atlas_map.topics.group_by_topic(depth_level)
        for topic_info in topics_at_depth:
            short_desc = topic_info.get('topic_short_description')
            datum_ids = topic_info.get('datum_ids')
            topic_counts[(depth_level, short_desc)] = len(datum_ids)
    return topic_counts

def fetch_topic_models_geojson(project_id: str, projection_id: str, nomic_api_key: str):
    """
    Fetches the topic model clustering GeoJSON from Nomic Atlas API.
    """
    print(f"Fetching topic models GeoJSON for project_id={project_id}, projection_id={projection_id}...")
    api_url = f"https://api-atlas.nomic.ai/v1/project/{project_id}/index/projection/{projection_id}"
    response = requests.get(
        api_url,
        headers={"Authorization": f"Bearer {nomic_api_key}"}
    )
    response.raise_for_status()
    print("Successfully fetched topic models GeoJSON.")
    return response.json()

def print_topic_model_summary_statistics(features):
    """
    Calculates and prints summary statistics for the topic model features.
    """
    depth_counts = {}
    short_desc_lengths = []
    desc_lengths = []
    overall_unique_names_by_topic_field_level = {}
    unique_names_by_feature_depth = {}
    for feature in features:
        properties = feature.get('properties', {})
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
            short_desc_lengths.append(len(properties.get('topic_short_description', '') or ''))
        if 'topic_description' in properties:
            desc_lengths.append(len(properties.get('topic_description', '') or ''))

    print('---------------------------------')
    print("Nomic Atlas Topic Model summary statistics")
    for d, count in sorted(depth_counts.items()):
        print(f"    Features with depth {d}: {count}")    
    avg_short_desc_len = sum(short_desc_lengths) / len(short_desc_lengths) if short_desc_lengths else 0
    print(f"    Average 'topic_short_description' length: {avg_short_desc_len:.2f} characters (for {len(short_desc_lengths)} descriptions)")
    # print(f"    Overall unique names by topic_field_level: {overall_unique_names_by_topic_field_level}") # Optional: for debugging
    # print(f"    Unique names by actual feature depth: {unique_names_by_feature_depth}") # Optional: for debugging
    print('---------------------------------')

def topic_hierarchy_report(atlas_map, features, topic_datum_counts):
    """
    Generates a report of the parent-child relationships between depth 1 and depth 2 topics,
    including their keywords and datum counts.
    
    Returns a formatted string containing the hierarchy report.
    """
    report = "\n--- Nomic Atlas Topic Model Hierarchy (Depth 1 to Depth 2) ---\n"
    hierarchy = atlas_map.topics.hierarchy
    def get_keywords(topic_name_to_find, depth_to_find, features_list):
        for feature in features_list:
            props = feature.get('properties', {})
            if props.get('depth') == depth_to_find and props.get('topic_short_description') == topic_name_to_find:
                return props.get('topic_description', 'N/A')
        return 'N/A'
    for topic_key, sub_topics in hierarchy.items():
        topic_name, topic_depth = topic_key
        if topic_depth == 1:
            parent_keywords = get_keywords(topic_name, 1, features)
            parent_datum_count = topic_datum_counts.get((1, topic_name), "N/A")
            report += f"\nParent Topic (Depth 1): (Datums: {parent_datum_count})\n"
            report += f"  Keywords: {parent_keywords}\n"
            report += "  Child Topics (Depth 2):\n"
            for sub_topic_entry in sub_topics:
                child_keywords = get_keywords(sub_topic_entry, 2, features)
                child_datum_count = topic_datum_counts.get((2, sub_topic_entry), "N/A")
                report += f"    - (Datums: {child_datum_count})\n"
                report += f"      Keywords: {child_keywords}\n"
    return report

def display_topic_comparison(data_df, topics_df, indexed_field, num_samples=3):
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
            'Indexed Field': str(row.get(indexed_field, "N/A"))[:40] + "...",
            'NomicAtlasTopicDepth1': row.get('topic_depth_1', "N/A"),
            'NomicAtlasTopicDepth2': row.get('topic_depth_2', "N/A"),
            'ProposedClaudeTopic': "TODO",
            'GroundTruthTopic': row.get('Topic', "N/A"),
        })
    comparison_df = pd.DataFrame(comparison_data)
    column_order = ['ItemIndex', 'Indexed Field', 'GroundTruthTopic', 'NomicAtlasTopicDepth1', 'NomicAtlasTopicDepth2', 'ProposedClaudeTopic']
    return comparison_df[[col for col in column_order if col in comparison_df.columns]]

if __name__ == "__main__":

    # Get ground truth topic labels and Nomic Atlas Topic Model labels
    topic_datum_counts = {} 
    atlas_dataset = AtlasDataset('ai-policy-recommendations')
    indexed_field = 'Details'
    atlas_map = atlas_dataset.maps[0]
    data_df = atlas_map.data.df[[indexed_field, 'Topic']].copy()
    topics_df = atlas_map.topics.df.copy()
    topic_datum_counts = get_topic_datum_counts(atlas_map, depths_to_fetch=[1, 2])

    # Get Atlas topic model clustering geojson 
    # (without labels, trying to use Claude Sonnet to relabel so we can benchmark these topics against ground truth)
    project_id = "f2bba6e5-e3df-46ef-ad79-7362bd3c48f5"
    projection_id = "1abba3fa-b89d-40b3-a0e2-0cf52b2a7f13"
    topic_models_geojson = fetch_topic_models_geojson(project_id, projection_id, NOMIC_API_KEY)
    topic_model = topic_models_geojson.get('topic_models')[0]
    features = topic_model.get('features')
    print_topic_model_summary_statistics(features)
    report = topic_hierarchy_report(atlas_map, features, topic_datum_counts)
    print(report)

    anthropic_client = anthropic.Anthropic()

    topic_hierarchy_token_count = anthropic_client.messages.count_tokens(
        model="claude-3-7-sonnet-20250219",
        system="You are a scientist",
        messages=[{
            "role": "user",
            "content": report
        }],
    )

    print(topic_hierarchy_token_count.model_dump())

    comparison_result_df = display_topic_comparison(data_df, topics_df, indexed_field, num_samples=5)
    if comparison_result_df is not None:
        print("\n--- Topic Comparison DataFrame ---")
        print(comparison_result_df.to_string())