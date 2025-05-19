import os
import anthropic
from nomic import AtlasDataset
import pandas as pd 
import requests
import json

NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL_NAME = "claude-3-7-sonnet-20250219" 

def get_topic_datum_counts(atlas_map, depths_to_fetch=[1, 2]):
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

    

    # Define the structured output schema for topic hierarchy
    topic_hierarchy_schema = {
        "name": "topic_hierarchy",
        "description": "Create a structured topic hierarchy with short description labels",
        "input_schema": {
            "type": "object",
            "properties": {
                "parent_topics": {
                    "type": "array",
                    "description": "Array of 8 parent topics at depth 1",
                    "items": {
                        "type": "object",
                        "properties": {
                            "parent_id": {
                                "type": "integer",
                                "description": "ID of the parent topic (1-8)"
                            },
                            "parent_name": {
                                "type": "string",
                                "description": "Short descriptive name (5-20 chars) for the parent topic"
                            },
                            "parent_datum_count": {
                                "type": "integer",
                                "description": "Number of data points in this parent topic cluster"
                            },
                            "child_topics": {
                                "type": "array",
                                "description": "Array of exactly 8 child topics for this parent",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "child_id": {
                                            "type": "integer",
                                            "description": "ID of the child topic (1-8)"
                                        },
                                        "child_name": {
                                            "type": "string",
                                            "description": "Short descriptive name (5-20 chars) for the child topic"
                                        },
                                        "child_datum_count": {
                                            "type": "integer",
                                            "description": "Number of data points in this child topic cluster"
                                        }
                                    },
                                    "required": ["child_id", "child_name", "child_datum_count"]
                                }
                            }
                        },
                        "required": ["parent_id", "parent_name", "parent_datum_count", "child_topics"]
                    }
                }
            },
            "required": ["parent_topics"]
        }
    }

    # Use structured outputs with Claude (streaming)
    with anthropic_client.messages.stream(
        model="claude-3-7-sonnet-20250219",
        max_tokens=4000,
        temperature=0.7,
        system="You are a brilliant semiotic topologist. You are given a tree-shaped topic model in keyword form and must produce an equivalently shaped model in short description form (5-20 characters each). Never repeat yourself. You MUST maintain the same structure as the input hierarchy.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": report
                    }
                ]
            }
        ],
        tools=[topic_hierarchy_schema],
        tool_choice={"type": "tool", "name": "topic_hierarchy"}
    ) as stream:
        message = None
        tool_content = ""
        
        for chunk in stream:
            if chunk.type == "content_block_delta" and chunk.delta.type == "input_json_delta":
                tool_content += chunk.delta.partial_json
                print(chunk.delta.partial_json, end="", flush=True)
            elif chunk.type == "message_stop":
                message = chunk.message

    # # Print and parse the structured response
    # print("Claude's Structured Response:")
    # if hasattr(message, 'content') and message.content:
    #     for content in message.content:
    #         if content.type == 'tool_use':
    #             topic_hierarchy = json.loads(content.input)
    #             print(json.dumps(topic_hierarchy, indent=2))
                
    #             # You can now use this structured data as needed
    #             print("\n--- Formatted Topic Hierarchy ---")
    #             for parent in topic_hierarchy['parent_topics']:
    #                 print(f"\n## {parent['parent_name']} (Datums: {parent['parent_datum_count']})")
    #                 for child in parent['child_topics']:
    #                     print(f"- {child['child_name']} ({child['child_datum_count']})")
    # else:
    #     print("No valid structured output received")

    comparison_result_df = display_topic_comparison(data_df, topics_df, indexed_field, num_samples=5)
    if comparison_result_df is not None:
        print("\n--- Topic Comparison DataFrame ---")
        print(comparison_result_df.to_string())