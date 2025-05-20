import os
import anthropic
from nomic import AtlasDataset
import pandas as pd 
import requests
import argparse

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
    print('---------------------------------')

def topic_hierarchy_report(atlas_map, features, topic_datum_counts, max_depth=3):
    """
    Generates a report of the parent-child relationships between topics up to max_depth,
    including their keywords.
    
    Returns a formatted string containing the hierarchy report.
    """
    report = f"\n--- Nomic Atlas Topic Model Hierarchy (Up to Depth {max_depth}) ---\n"
    hierarchy = atlas_map.topics.hierarchy 
    keyword_lookup = {}
    meta_df = atlas_map.topics.metadata
    for _, row in meta_df.iterrows():
        depth = row.get('depth')
        short_desc = str(row.get('topic_short_description', '')).strip()
        topic_desc_keywords = str(row.get('topic_description', 'N/A'))
        if depth is not None and short_desc and int(depth) <= max_depth:
            keyword_lookup[(int(depth), short_desc)] = topic_desc_keywords

    processed_parents = set()
    sorted_hierarchy_items = sorted(hierarchy.items(), key=lambda item: (int(item[0][1]), str(item[0][0]).strip()))
    for nomic_parent_key_from_hierarchy, nomic_children_names_list in sorted_hierarchy_items:
        nomic_parent_name_str = str(nomic_parent_key_from_hierarchy[0]).strip()
        nomic_parent_depth_int = int(nomic_parent_key_from_hierarchy[1])
        actual_lookup_key = (nomic_parent_depth_int, nomic_parent_name_str) 
        if nomic_parent_depth_int == 1 and actual_lookup_key not in processed_parents:
            parent_keywords = keyword_lookup.get(actual_lookup_key, 'N/A') 
            report += f"\nParent Topic (Depth 1):\n"
            report += f"  Keywords: {parent_keywords}\n"
            processed_parents.add(actual_lookup_key)
            if nomic_children_names_list and max_depth >= 2:
                report += "  Child Topics (Depth 2):\n"
                for nomic_child_name_from_hierarchy in sorted(nomic_children_names_list):
                    nomic_child_name_str = str(nomic_child_name_from_hierarchy).strip()
                    child_actual_lookup_key = (2, nomic_child_name_str) 
                    child_keywords = keyword_lookup.get(child_actual_lookup_key, 'N/A')
                    report += f"    - Keywords: {child_keywords}\n"
                    nomic_grandchildren_names_list = hierarchy.get((nomic_child_name_from_hierarchy, 2), [])
                    if nomic_grandchildren_names_list and max_depth >= 3:
                        report += "      Grandchild Topics (Depth 3):\n"
                        for nomic_grandchild_name_from_hierarchy in sorted(nomic_grandchildren_names_list):
                            nomic_grandchild_name_str = str(nomic_grandchild_name_from_hierarchy).strip()
                            grandchild_actual_lookup_key = (3, nomic_grandchild_name_str) 
                            grandchild_keywords = keyword_lookup.get(grandchild_actual_lookup_key, 'N/A')
                            report += f"        - Keywords: {grandchild_keywords}\n"
    return report

def create_topic_mapping(atlas_map, claude_topic_hierarchy):
    """
    Creates a mapping from Nomic Atlas topic keywords to Claude-generated topic names.
    """
    depth1_mapping = {}
    depth2_mapping = {}
    depth3_mapping = {} 
    
    # Get the ordered list of Nomic topics from the hierarchy
    hierarchy = atlas_map.topics.hierarchy
    nomic_depth1_topics_ordered = sorted([key[0] for key in hierarchy.keys() if key[1] == 1]) # Assuming consistent ordering is desired

    # Pre-index Claude parent topics by name for faster lookup
    claude_parents_by_name = {}
    if claude_topic_hierarchy and 'parent_topics' in claude_topic_hierarchy:
        claude_parents_by_name = {p['parent_name']: p for p in claude_topic_hierarchy['parent_topics']}
    
    # Create depth 1 mapping (Nomic short_desc -> Claude name)
    if claude_topic_hierarchy and 'parent_topics' in claude_topic_hierarchy:
        for i, claude_parent_topic_obj in enumerate(claude_topic_hierarchy['parent_topics']):
            if i < len(nomic_depth1_topics_ordered):
                nomic_topic_short_desc = nomic_depth1_topics_ordered[i]
                claude_topic_name = claude_parent_topic_obj['parent_name']
                depth1_mapping[nomic_topic_short_desc] = claude_topic_name
    
    # Create depth 2 and 3 mappings
    # Iterate through Nomic hierarchy for D1 parents to ensure we cover all Nomic branches
    for nomic_parent_short_desc, nomic_parent_depth in hierarchy.keys():
        if nomic_parent_depth == 1 and nomic_parent_short_desc in depth1_mapping:
            claude_parent_name = depth1_mapping[nomic_parent_short_desc]
            claude_parent_obj = claude_parents_by_name.get(claude_parent_name)
            nomic_child_short_descs_list = hierarchy.get((nomic_parent_short_desc, 1), [])
            if claude_parent_obj and 'child_topics' in claude_parent_obj and nomic_child_short_descs_list:
                for i, nomic_child_short_desc in enumerate(nomic_child_short_descs_list):
                    if i < len(claude_parent_obj['child_topics']):
                        claude_child_obj = claude_parent_obj['child_topics'][i]
                        claude_child_name = claude_child_obj['child_name']
                        depth2_mapping[(nomic_parent_short_desc, nomic_child_short_desc)] = claude_child_name
                        
                        nomic_grandchildren_short_descs_list = hierarchy.get((nomic_child_short_desc, 2), [])
                        if nomic_grandchildren_short_descs_list and 'grandchild_topics' in claude_child_obj:
                            for j, nomic_grandchild_short_desc in enumerate(nomic_grandchildren_short_descs_list):
                                if j < len(claude_child_obj['grandchild_topics']):
                                    claude_grandchild_name = claude_child_obj['grandchild_topics'][j]['grandchild_name']
                                    depth3_mapping[(nomic_parent_short_desc, nomic_child_short_desc, nomic_grandchild_short_desc)] = claude_grandchild_name
    
    return depth1_mapping, depth2_mapping, depth3_mapping

def topic_comparison(data_df, topics_df, indexed_field, claude_depth1_mapping=None, claude_depth2_mapping=None, claude_depth3_mapping=None):
    """
    Constructs and returns a DataFrame comparing Nomic Atlas topics vs Nomic Atlas + Claude Sonnet topics.
    """
    cols_to_join = ['topic_depth_1', 'topic_depth_2', 'topic_depth_3']
    topics_df_subset = topics_df[[col for col in cols_to_join if col in topics_df.columns]]
    merged_df = data_df.join(topics_df_subset, how='left') 
    comparison_data = []    
    for index, row in merged_df.iterrows():
        nomic_depth1 = row.get('topic_depth_1')
        nomic_depth2 = row.get('topic_depth_2')
        nomic_depth3 = row.get('topic_depth_3')
        claude_depth1 = claude_depth1_mapping.get(nomic_depth1)
        claude_depth2 = claude_depth2_mapping.get((nomic_depth1, nomic_depth2))
        claude_depth3 = claude_depth3_mapping.get((nomic_depth1, nomic_depth2, nomic_depth3))
        comparison_data.append({
            'Datum ID': index,
            'Indexed Field': row.get(indexed_field),
            'NomicAtlasTopicDepth1': nomic_depth1,
            'NomicAtlasTopicDepth2': nomic_depth2,
            'NomicAtlasTopicDepth3': nomic_depth3,
            'NomicClaudeTopicDepth1': claude_depth1,
            'NomicClaudeTopicDepth2': claude_depth2,
            'NomicClaudeTopicDepth3': claude_depth3,
        })
    return pd.DataFrame(comparison_data)

def get_topic_hierarchy_from_claude(anthropic_client, report, actual_max_depth):
    
    base_child_properties = {
        "child_id": {
            "type": "integer",
            "description": "ID of the child topic"
        },
        "child_name": {
            "type": "string",
            "description": "Short descriptive name for the child topic. SUPER IMPORTANT: MUST BE 5-20 CHARACTERS."
        }
    }
    required_child_fields = ["child_id", "child_name"]

    # Dynamically add grandchild_topics if actual_max_depth supports
    assert actual_max_depth <= 3, "Actual max depth must be <= 3"
    if actual_max_depth == 3:
        base_child_properties["grandchild_topics"] = {
            "type": "array",
            "description": "Array of grandchild topics for this child (variable number)",
            "items": {
                "type": "object",
                "properties": {
                    "grandchild_id": {
                        "type": "integer",
                        "description": "ID of the grandchild topic"
                    },
                    "grandchild_name": {
                        "type": "string",
                        "description": "Short descriptive name for the grandchild topic. SUPER IMPORTANT: MUST BE 5-20 CHARACTERS."
                    }
                },
                "required": ["grandchild_id", "grandchild_name"]
            }
        }
        required_child_fields.append("grandchild_topics")

    topic_hierarchy_schema = {
        "name": "topic_hierarchy",
        "description": "Create a structured topic hierarchy with short description labels",
        "input_schema": {
            "type": "object",
            "properties": {
                "parent_topics": {
                    "type": "array",
                    "description": "Array of parent topics at depth 1 (typically 8)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "parent_id": {
                                "type": "integer",
                                "description": "ID of the parent topic"
                            },
                            "parent_name": {
                                "type": "string",
                                "description": "Short descriptive name for the parent topic. SUPER IMPORTANT: MUST BE 5-20 CHARACTERS."
                            },
                            "child_topics": {
                                "type": "array",
                                "description": "Array of child topics for this parent (variable number)",
                                "items": {
                                    "type": "object",
                                    "properties": base_child_properties, # Use dynamically built child properties
                                    "required": required_child_fields # Use dynamically built required fields
                                }
                            }
                        },
                        "required": ["parent_id", "parent_name", "child_topics"]
                    }
                }
            },
            "required": ["parent_topics"]
        }
    }

    with anthropic_client.messages.stream(
        model=CLAUDE_MODEL_NAME,
        max_tokens=40000, 
        temperature=0.2,
        system='''You are a topological semiotician specializing in structured hierarchical ontology optimization.
You are given a tree-shaped topic model (defined by keywords from a Nomic Atlas map) and your task is to generate short, descriptive names (5-20 characters each) for each topic node (parent, child, grandchild).
Crucially, your output MUST preserve the exact hierarchical structure (parent-child-grandchild relationships) of the input Nomic topic model. You are labeling the existing Nomic structure, not creating a new one.

Your primary goals for the descriptive names are:
1.  **Strict Hierarchical Validity**:
- Each child topic\'s name MUST represent a clear, logical, and intuitive conceptual subset or a specific aspect of its parent topic\'s name.
- Similarly, each grandchild topic\'s name MUST be a conceptual subset or specific aspect of its direct child topic parent\'s name.
- The parent-child-grandchild relationships in your names must be directionally coherent (e.g., Parent: "Vehicles", Child: "Cars", Grandchild: "Sedans").

2.  **Maximal Conceptual Distinctness & Non-Redundancy in Labeling**:
- Each parent topic name must define a distinct conceptual area based on its keywords.
- Within a family, each child topic name must be unique and clearly differentiate its concept from its siblings, based on their respective keywords.
- Across the entire hierarchy: Strive for unique names for unique Nomic topic nodes. If different Nomic nodes (e.g., under different parents or at different levels) have keywords pointing to similar underlying real-world themes, your descriptive names for these *specific Nomic nodes* must be nuanced and distinct. Aim for names that reflect the particular keywords and the specific parental context of each Nomic node. Avoid using the exact same name for two different topic nodes unless their underlying Nomic keywords and hierarchical position are identical.

3.  **Accuracy to Nomic Keywords**:
- Each descriptive name must be a concise and accurate summary of the keywords associated with its corresponding Nomic topic node. The name should be directly derivable from the provided keywords for that node.

4.  **Clarity and Conciseness**:
- Names must be 5-20 characters long.
- Use full words when possible. Only use widely recognized and unambiguous abbreviations (e.g., "USA", "AI", "UK"). Avoid creating non-standard or obscure abbreviations to meet character limits.
- Focus on meaningful, easily understandable terms.

Recap of Critical Instructions:
- Adhere strictly to the provided Nomic topic structure. Your role is to label, not restructure.
- Ensure all names create a strong, intuitive, and valid conceptual hierarchy (e.g., child is a specific type of parent).
- Maximize label distinctiveness across all topic nodes, using context and specific keywords to differentiate.
- Names must accurately reflect the given Nomic keywords for each node and respect character limits.
- Prioritize clarity and standard language over forced abbreviations.
''',
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
        print("Streaming response from Claude...")
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "input_json_delta":
                    print(event.delta.partial_json, end="", flush=True)
            elif event.type == "message_stop":
                print("\nStream finished.")
        
        final_message = stream.get_final_message()
    
    return final_message.content[0].input


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process Nomic Atlas Topic Model with Claude")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--indexed-field", type=str)
    args = parser.parse_args()

    # Get Nomic Atlas Topic Model labels
    topic_datum_counts = {} 
    atlas_dataset = AtlasDataset(args.dataset)
    indexed_field = args.indexed_field
    atlas_map = atlas_dataset.maps[0]

    # Determine the actual max depth from topic metadata
    actual_max_depth = 0
    meta_df = atlas_map.topics.metadata
    actual_max_depth = meta_df['depth'].max()    
    depths_to_fetch = list(range(1, int(actual_max_depth) + 1))
    data_df = atlas_map.data.df[[indexed_field]].copy()
    topics_df = atlas_map.topics.df.copy()
    topic_datum_counts = get_topic_datum_counts(atlas_map, depths_to_fetch=depths_to_fetch)

    # Get Atlas topic model clustering geojson 
    project_id = atlas_dataset.id
    projection_id = atlas_dataset.maps[0].projection_id
    topic_models_geojson = fetch_topic_models_geojson(project_id, projection_id, NOMIC_API_KEY)
    topic_model = topic_models_geojson.get('topic_models')[0]
    features = topic_model.get('features')
    print_topic_model_summary_statistics(features)
    report = topic_hierarchy_report(atlas_map, features, topic_datum_counts, max_depth=actual_max_depth)
    print(report)

    anthropic_client = anthropic.Anthropic()
    token_count = anthropic_client.messages.count_tokens(
        model="claude-3-7-sonnet-20250219",
        system="You are a scientist",
        messages=[{
            "role": "user",
            "content": report
        }],
    )
    print("quick token count:")
    print(token_count.model_dump())
    
    # Get topic hierarchy from Claude Sonnet
    claude_topic_hierarchy = get_topic_hierarchy_from_claude(anthropic_client, report, actual_max_depth)
    print("--------------------------------")
    depth1_mapping, depth2_mapping, depth3_mapping = create_topic_mapping(atlas_map, claude_topic_hierarchy)
    output_lines = ["\n--- Formatted Topic Hierarchy ---"]
    if claude_topic_hierarchy and 'parent_topics' in claude_topic_hierarchy:
        for parent in claude_topic_hierarchy['parent_topics']:
            output_lines.append(f"\n## {parent['parent_name']}")
            if 'child_topics' in parent:
                for child in parent['child_topics']:
                    output_lines.append(f"- {child['child_name']}")
                    if 'grandchild_topics' in child and child['grandchild_topics']:
                        for grandchild in child['grandchild_topics']:
                            output_lines.append(f"  * {grandchild['grandchild_name']}")
    print("\n".join(output_lines))

    # Compare original method for Nomic Atlas topics vs Nomic Atlas + Claude Sonnet topics
    comparison_result_df = topic_comparison(data_df, topics_df, indexed_field, claude_depth1_mapping=depth1_mapping, claude_depth2_mapping=depth2_mapping, claude_depth3_mapping=depth3_mapping)            
    comparison_result_df.to_csv(f'{args.dataset}_topic_comparison.csv', index=False)
    