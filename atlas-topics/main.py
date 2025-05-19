import os
import anthropic
from nomic import AtlasDataset
import pandas as pd 
import requests

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

def create_topic_mapping(atlas_map, claude_topic_hierarchy):
    """
    Creates a mapping from Nomic Atlas topic keywords to Claude-generated topic names.
    Returns two dictionaries - one for depth 1 and one for depth 2 mappings.
    """
    depth1_mapping = {}
    depth2_mapping = {}
    
    # Get the ordered list of topics from the hierarchy
    hierarchy = atlas_map.topics.hierarchy
    nomic_depth1_topics = [topic_key[0] for topic_key in hierarchy.keys() if topic_key[1] == 1]
    
    # Create depth 1 mapping
    for i, parent_topic in enumerate(claude_topic_hierarchy['parent_topics']):
        if i < len(nomic_depth1_topics):
            nomic_topic = nomic_depth1_topics[i]
            claude_topic = parent_topic['parent_name']
            depth1_mapping[nomic_topic] = claude_topic
    
    # Create depth 2 mapping
    for parent_key, child_topics in hierarchy.items():
        parent_name, parent_depth = parent_key
        if parent_depth == 1 and parent_name in depth1_mapping:
            # Find corresponding Claude parent
            claude_parent_name = depth1_mapping[parent_name]
            claude_parent = next((p for p in claude_topic_hierarchy['parent_topics'] if p['parent_name'] == claude_parent_name), None)
            if claude_parent and len(child_topics) > 0:
                # Map child topics in order
                for i, nomic_child in enumerate(child_topics):
                    if i < len(claude_parent['child_topics']):
                        claude_child = claude_parent['child_topics'][i]['child_name']
                        depth2_mapping[(parent_name, nomic_child)] = claude_child
    
    return depth1_mapping, depth2_mapping

def topic_comparison(data_df, topics_df, indexed_field, claude_depth1_mapping=None, claude_depth2_mapping=None):
    """
    Constructs and returns a DataFrame comparing ground truth topics, Nomic Atlas topics,
    and Claude Sonnet topics.
    """
    cols_to_join = ['topic_depth_1', 'topic_depth_2']
    topics_df_subset = topics_df[[col for col in cols_to_join if col in topics_df.columns]]
    merged_df = data_df.join(topics_df_subset, how='left') 
    # sample_df = merged_df.sample(min(num_samples, len(merged_df)))
    comparison_data = []
    
    for index, row in merged_df.iterrows():
        nomic_depth1 = row.get('topic_depth_1', "N/A")
        nomic_depth2 = row.get('topic_depth_2', "N/A")
        claude_depth1 = claude_depth1_mapping.get(nomic_depth1, "N/A")
        claude_depth2 = claude_depth2_mapping.get((nomic_depth1, nomic_depth2), "N/A")
        comparison_data.append({
            'Datum ID': index,
            'Indexed Field': row.get(indexed_field, "N/A"),
            'NomicAtlasTopicDepth1': nomic_depth1,
            'NomicAtlasTopicDepth2': nomic_depth2,
            'NomicClaudeTopicDepth1': claude_depth1,
            'NomicClaudeTopicDepth2': claude_depth2,
            'GroundTruthTopic': row.get('Topic', "N/A"),
        })
    return pd.DataFrame(comparison_data)

def get_topic_hierarchy_from_claude(report):
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
                                "description": "Short descriptive name for the parent topic. SUPER IMPORTANT: MUST BE 5-20 CHARACTERS."
                            },
                            "parent_datum_count": {
                                "type": "integer",
                                "description": "Number of data points in this parent topic cluster"
                            },
                            "child_topics": {
                                "type": "array",
                                "description": "Array of child topics for this parent (variable number)",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "child_id": {
                                            "type": "integer",
                                            "description": "ID of the child topic"
                                        },
                                        "child_name": {
                                            "type": "string",
                                            "description": "Short descriptive name for the child topic. SUPER IMPORTANT: MUST BE 5-20 CHARACTERS."
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
    message = anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=4000,
        temperature=0.7,
        system="""You are a topological semiotician specializing in structured hierarhical ontology optimization.
You are given a tree-shaped topic model in keyword form and must produce an equivalently shaped model in short description form (5-20 characters each).
Your primary goal is to REDUCE REDUNDANCY while maintaining the hierarchical structure:

1. Ensure each parent topic has a distinct conceptual focus that doesn't overlap significantly with other parent topics
2. Each child topic should have a clear, unique relationship to its parent topic that differentiates it from children of other parents
3. Use distinct terminology across topics to avoid semantic overlap (e.g., don't use 'Medical AI' under multiple parent categories)
4. When topics appear conceptually similar, use more specific terms to highlight their unique aspects
5. Strive for orthogonal topic dimensions where possible, minimizing cross-cutting concerns

You maintain the SAME STRUCTURE of parent child topics as the input hierarchy, but make each topic description conceptually distinct while preserving accurate representation.""",
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
    )
    return message.content[0].input

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

    # Get topic hierarchy from Claude Sonnet
    claude_topic_hierarchy = get_topic_hierarchy_from_claude(report)
    depth1_mapping, depth2_mapping = create_topic_mapping(atlas_map, claude_topic_hierarchy)
    print("\n--- Formatted Topic Hierarchy ---")
    for parent in claude_topic_hierarchy['parent_topics']:
        print(f"\n## {parent['parent_name']} (Datums: {parent['parent_datum_count']})")
        for child in parent['child_topics']:
            print(f"- {child['child_name']} ({child['child_datum_count']})")

    # Compare ground truth topics, Nomic Atlas topics, and Claude Sonnet topics
    comparison_result_df = topic_comparison(data_df, topics_df, indexed_field, claude_depth1_mapping=depth1_mapping, claude_depth2_mapping=depth2_mapping)            
    print("\n--- Topic Comparison DataFrame ---")
    print(comparison_result_df.to_string())
    comparison_result_df.to_csv('comparison_result_df.csv', index=False)