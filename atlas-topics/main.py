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
    including their keywords and Nomic topic_ids.
    
    Returns a formatted string containing the hierarchy report.
    """
    report = f"\n--- Nomic Atlas Topic Model Hierarchy (Up to Depth {max_depth}) ---\n"
    hierarchy = atlas_map.topics.hierarchy 
    keyword_lookup = {}
    meta_df = atlas_map.topics.metadata
    
    # Create a lookup for (depth, topic_short_description) -> topic_id
    desc_to_id_lookup = {}
    if not meta_df.empty:
        for _, row in meta_df.iterrows():
            depth_val = int(row['depth'])
            short_desc_val = str(row.get('topic_short_description', '')).strip()
            topic_id_val = int(row['topic_id'])
            if short_desc_val: # Ensure not empty
                desc_to_id_lookup[(depth_val, short_desc_val)] = topic_id_val

    for _, row in meta_df.iterrows():
        depth = row.get('depth')
        short_desc = str(row.get('topic_short_description', '')).strip()
        topic_desc_keywords = str(row.get('topic_description', 'N/A'))
        if depth is not None and short_desc and int(depth) <= max_depth:
            keyword_lookup[(int(depth), short_desc)] = topic_desc_keywords

    processed_parents = set()
    # Sort by depth, then by Nomic ID if available, then by name, to ensure consistent order for Claude
    # This requires fetching Nomic IDs for sorting keys before iterating hierarchy.items()
    
    hierarchy_items_with_ids_for_sorting = []
    for nomic_parent_key, _ in hierarchy.items():
        name_str, depth_int = str(nomic_parent_key[0]).strip(), int(nomic_parent_key[1])
        nomic_id = desc_to_id_lookup.get((depth_int, name_str), -1)
        hierarchy_items_with_ids_for_sorting.append(((name_str, depth_int, nomic_id), nomic_parent_key))

    sorted_hierarchy_keys = [
        item[1] for item in sorted(hierarchy_items_with_ids_for_sorting, key=lambda x: (x[0][1], x[0][2], x[0][0]))
    ]

    for nomic_parent_key_from_hierarchy in sorted_hierarchy_keys:
        nomic_children_names_list = hierarchy[nomic_parent_key_from_hierarchy]
        nomic_parent_name_str = str(nomic_parent_key_from_hierarchy[0]).strip()
        nomic_parent_depth_int = int(nomic_parent_key_from_hierarchy[1])
        
        parent_nomic_id = desc_to_id_lookup.get((nomic_parent_depth_int, nomic_parent_name_str), -1)
        actual_lookup_key = (nomic_parent_depth_int, nomic_parent_name_str) 
        
        if nomic_parent_depth_int == 1 and actual_lookup_key not in processed_parents:
            parent_keywords = keyword_lookup.get(actual_lookup_key, 'N/A') 
            report += f"\nParent (ID: {parent_nomic_id}) {parent_keywords}\n"
            processed_parents.add(actual_lookup_key)
            
            if nomic_children_names_list and max_depth >= 2:
                report += "  Child:\n"
                # Sort children by their Nomic ID then name for consistent order
                sorted_children = []
                for child_name in nomic_children_names_list:
                    child_id = desc_to_id_lookup.get((2, str(child_name).strip()), -1)
                    sorted_children.append((str(child_name).strip(), child_id))
                
                for nomic_child_name_str, child_nomic_id in sorted(sorted_children, key=lambda x: (x[1], x[0])):
                    child_actual_lookup_key = (2, nomic_child_name_str) 
                    child_keywords = keyword_lookup.get(child_actual_lookup_key, 'N/A')
                    report += f"    - (ID: {child_nomic_id}) {child_keywords}\n"
                    
                    # Children from hierarchy are just names, depth is parent_depth + 1 (i.e. 2 for these children)
                    # Grandchildren are looked up using (child_name_from_hierarchy, 2) as key
                    nomic_grandchildren_names_list = hierarchy.get((nomic_child_name_str, 2), [])
                    if nomic_grandchildren_names_list and max_depth >= 3:
                        report += "      Grandchild:\n"
                        # Sort grandchildren by their Nomic ID then name
                        sorted_grandchildren = []
                        for grandchild_name in nomic_grandchildren_names_list:
                            grandchild_id = desc_to_id_lookup.get((3, str(grandchild_name).strip()), -1)
                            sorted_grandchildren.append((str(grandchild_name).strip(), grandchild_id))
                        
                        for nomic_grandchild_name_str, grandchild_nomic_id in sorted(sorted_grandchildren, key=lambda x: (x[1], x[0])):
                            grandchild_actual_lookup_key = (3, nomic_grandchild_name_str) 
                            grandchild_keywords = keyword_lookup.get(grandchild_actual_lookup_key, 'N/A')
                            report += f"        - (ID: {grandchild_nomic_id}) {grandchild_keywords}\n"
    return report

def validate_topic_hierarchy_completeness(hierarchy, actual_max_depth):
    """Quick validation to check if all topics have names and IDs"""
    if 'parent_topics' not in hierarchy:
        print("Validation Error: 'parent_topics' key missing from Claude response.")
        return False
    
    missing_data = False
    
    for parent in hierarchy['parent_topics']:
        if not isinstance(parent.get('parent_id'), int) or not parent.get('parent_name'):
            print(f"Validation Error: Parent missing ID or name: {parent}")
            missing_data = True
        
        if 'child_topics' not in parent:
            print(f"Validation Error: Parent {parent.get('parent_name', '(ID: '+str(parent.get('parent_id'))+')')} missing 'child_topics' array.")
            missing_data = True # If child_topics is missing, we can't check children
            continue 
            
        for child in parent['child_topics']:
            if not isinstance(child.get('child_id'), int) or not child.get('child_name'):
                print(f"Validation Error: Child missing ID or name under parent {parent.get('parent_name', '(ID: '+str(parent.get('parent_id'))+')')}: {child}")
                missing_data = True
            
            if actual_max_depth == 3:
                if 'grandchild_topics' not in child:
                    print(f"Validation Error: Child {child.get('child_name', '(ID: '+str(child.get('child_id'))+')')} missing 'grandchild_topics' array when depth 3 expected.")
                    missing_data = True # If grandchild_topics is missing, can't check grandchildren
                    continue
                    
                for grandchild in child['grandchild_topics']:
                    if not isinstance(grandchild.get('grandchild_id'), int) or not grandchild.get('grandchild_name'):
                        print(f"Validation Error: Grandchild missing ID or name under child {child.get('child_name', '(ID: '+str(child.get('child_id'))+')')}: {grandchild}")
                        missing_data = True
    
    return not missing_data

def create_topic_mapping(atlas_map, claude_topic_hierarchy):
    """
    Creates a mapping from Nomic Atlas topic_ids to Claude-generated topic names.
    """
    depth1_mapping = {} # Nomic D1 topic_id -> Claude Name
    depth2_mapping = {} # (Nomic D1 topic_id, Nomic D2 topic_id) -> Claude Name
    depth3_mapping = {} # (Nomic D1 topic_id, Nomic D2 topic_id, Nomic D3 topic_id) -> Claude Name

    # For validation and error reporting if Claude returns an ID not in metadata
    valid_nomic_topic_ids = set(atlas_map.topics.metadata['topic_id'].astype(int).unique())

    # For reporting coverage against original Nomic topic counts per depth
    meta_df = atlas_map.topics.metadata
    nomic_topic_ids_by_depth = {
        1: set(meta_df[meta_df['depth'] == 1]['topic_id'].astype(int).unique()),
        2: set(meta_df[meta_df['depth'] == 2]['topic_id'].astype(int).unique()),
        3: set(meta_df[meta_df['depth'] == 3]['topic_id'].astype(int).unique())
    }
    total_nomic_depth1 = len(nomic_topic_ids_by_depth[1])
    total_nomic_depth2 = len(nomic_topic_ids_by_depth[2])
    total_nomic_depth3 = len(nomic_topic_ids_by_depth[3])
    
    mapped_nomic_ids_depth1 = set()
    mapped_nomic_ids_depth2 = set()
    mapped_nomic_ids_depth3 = set()

    if not claude_topic_hierarchy or 'parent_topics' not in claude_topic_hierarchy:
        print("ERROR: Claude topic hierarchy is empty or invalid in create_topic_mapping.")
        return depth1_mapping, depth2_mapping, depth3_mapping

    for claude_parent in claude_topic_hierarchy.get('parent_topics', []):
        parent_nomic_id = claude_parent.get('parent_id')
        claude_parent_name = claude_parent.get('parent_name')
        
        if parent_nomic_id is None or claude_parent_name is None:
            print(f"WARNING: Claude parent missing ID or name: {claude_parent}")
            continue
        if parent_nomic_id not in valid_nomic_topic_ids:
            print(f"WARNING: Claude parent_id {parent_nomic_id} not found in Nomic metadata.")
            continue
            
        depth1_mapping[parent_nomic_id] = claude_parent_name
        if parent_nomic_id in nomic_topic_ids_by_depth[1]:
            mapped_nomic_ids_depth1.add(parent_nomic_id)

        for claude_child in claude_parent.get('child_topics', []):
            child_nomic_id = claude_child.get('child_id')
            claude_child_name = claude_child.get('child_name')

            if child_nomic_id is None or claude_child_name is None:
                print(f"WARNING: Claude child missing ID or name under parent ID {parent_nomic_id}: {claude_child}")
                continue
            if child_nomic_id not in valid_nomic_topic_ids:
                print(f"WARNING: Claude child_id {child_nomic_id} not found in Nomic metadata.")
                continue
            
            depth2_mapping[(parent_nomic_id, child_nomic_id)] = claude_child_name
            if child_nomic_id in nomic_topic_ids_by_depth[2]:
                 mapped_nomic_ids_depth2.add(child_nomic_id) # Count unique D2 IDs mapped

            for claude_grandchild in claude_child.get('grandchild_topics', []):
                grandchild_nomic_id = claude_grandchild.get('grandchild_id')
                claude_grandchild_name = claude_grandchild.get('grandchild_name')
                
                if grandchild_nomic_id is None or claude_grandchild_name is None:
                    print(f"WARNING: Claude grandchild missing ID or name under child ID {child_nomic_id}: {claude_grandchild}")
                    continue
                if grandchild_nomic_id not in valid_nomic_topic_ids:
                    print(f"WARNING: Claude grandchild_id {grandchild_nomic_id} not found in Nomic metadata.")
                    continue
                
                depth3_mapping[(parent_nomic_id, child_nomic_id, grandchild_nomic_id)] = claude_grandchild_name
                if grandchild_nomic_id in nomic_topic_ids_by_depth[3]:
                    mapped_nomic_ids_depth3.add(grandchild_nomic_id) # Count unique D3 IDs mapped
    
    print("\nTopic Mapping Coverage (Based on unique Nomic topic_ids mapped by Claude):")
    print(f"Depth 1: {len(mapped_nomic_ids_depth1)}/{total_nomic_depth1} Nomic D1 topics mapped ({ (len(mapped_nomic_ids_depth1)/total_nomic_depth1*100) if total_nomic_depth1 else 0 :.1f}%)")
    print(f"Depth 2: {len(mapped_nomic_ids_depth2)}/{total_nomic_depth2} Nomic D2 topics mapped ({ (len(mapped_nomic_ids_depth2)/total_nomic_depth2*100) if total_nomic_depth2 else 0 :.1f}%)")
    if total_nomic_depth3 > 0:
        print(f"Depth 3: {len(mapped_nomic_ids_depth3)}/{total_nomic_depth3} Nomic D3 topics mapped ({ (len(mapped_nomic_ids_depth3)/total_nomic_depth3*100) if total_nomic_depth3 else 0 :.1f}%)")

    # (Claude node count comparison warning can remain as is, it's a useful structural check)
    claude_node_counts = {'depth1': 0, 'depth2': 0, 'depth3': 0}
    if claude_topic_hierarchy and 'parent_topics' in claude_topic_hierarchy:
        claude_node_counts['depth1'] = len(claude_topic_hierarchy['parent_topics'])
        for p in claude_topic_hierarchy['parent_topics']:
            claude_node_counts['depth2'] += len(p.get('child_topics', []))
            for c in p.get('child_topics', []):
                claude_node_counts['depth3'] += len(c.get('grandchild_topics', []))
    
    if claude_node_counts['depth1'] != total_nomic_depth1:
        print(f"STRUCTURAL WARNING: Claude returned {claude_node_counts['depth1']} parent topics, Nomic had {total_nomic_depth1}")
    if claude_node_counts['depth2'] != total_nomic_depth2:
        print(f"STRUCTURAL WARNING: Claude returned {claude_node_counts['depth2']} child topics, Nomic had {total_nomic_depth2}")
    if total_nomic_depth3 > 0 and claude_node_counts['depth3'] != total_nomic_depth3:
         print(f"STRUCTURAL WARNING: Claude returned {claude_node_counts['depth3']} grandchild topics, Nomic had {total_nomic_depth3}")

    return depth1_mapping, depth2_mapping, depth3_mapping

def topic_comparison(data_df, topics_df, indexed_field, atlas_map, claude_depth1_mapping=None, claude_depth2_mapping=None, claude_depth3_mapping=None):
    """
    Constructs and returns a DataFrame comparing Nomic Atlas topics vs Nomic Atlas + Claude Sonnet topics.
    Uses Nomic topic_ids for robust lookup into Claude mappings.
    Reports on unique Nomic topic_ids that are missing Claude mappings.
    """
    cols_to_join = ['topic_depth_1', 'topic_depth_2', 'topic_depth_3']
    topics_df_cols_to_select = [col for col in cols_to_join if col in topics_df.columns]
    topics_df_subset = topics_df[topics_df_cols_to_select]
    merged_df = data_df.join(topics_df_subset, how='left')
    comparison_data = []

    # Create a lookup: (depth, stripped_topic_short_description) -> topic_id
    # This is crucial for getting the correct topic_id from the names in topics.df
    desc_to_id_lookup = {}
    if hasattr(atlas_map, 'topics') and hasattr(atlas_map.topics, 'metadata'):
        for _, row in atlas_map.topics.metadata.iterrows():
            depth = int(row['depth'])
            short_desc = str(row.get('topic_short_description', '')).strip()
            topic_id = int(row['topic_id'])
            if short_desc: # Ensure not empty
                desc_to_id_lookup[(depth, short_desc)] = topic_id
    else:
        print("ERROR: atlas_map.topics.metadata not available in topic_comparison. Cannot map names to IDs.")
        # Fallback or error handling if metadata is not available
        # For now, we'll proceed, but lookups will likely fail if this happens.

    missing_nomic_topic_ids_depth1 = set()
    missing_nomic_topic_ids_depth2 = set() # Stores (parent_id, child_id) tuples
    missing_nomic_topic_ids_depth3 = set() # Stores (parent_id, child_id, grandchild_id) tuples
    
    printed_d1_debug_info = False

    for index, row in merged_df.iterrows():
        nomic_d1_name_raw = row.get('topic_depth_1')
        nomic_d2_name_raw = row.get('topic_depth_2')
        nomic_d3_name_raw = row.get('topic_depth_3')

        nomic_d1_name_stripped = str(nomic_d1_name_raw).strip() if pd.notna(nomic_d1_name_raw) else None
        nomic_d2_name_stripped = str(nomic_d2_name_raw).strip() if pd.notna(nomic_d2_name_raw) else None
        nomic_d3_name_stripped = str(nomic_d3_name_raw).strip() if pd.notna(nomic_d3_name_raw) else None

        nomic_d1_id, nomic_d2_id, nomic_d3_id = None, None, None
        claude_depth1, claude_depth2, claude_depth3 = None, None, None

        if nomic_d1_name_stripped:
            nomic_d1_id = desc_to_id_lookup.get((1, nomic_d1_name_stripped))
            if nomic_d1_id is not None:
                claude_depth1 = claude_depth1_mapping.get(nomic_d1_id)
                if not claude_depth1:
                    missing_nomic_topic_ids_depth1.add(nomic_d1_id)
                    # (Diagnostic print for first failed D1 lookup can be re-added here if needed)
            elif nomic_d1_name_stripped: # Name was there, but no ID found in metadata
                 missing_nomic_topic_ids_depth1.add(f"NameNotInMeta:D1:{nomic_d1_name_stripped}")

        if nomic_d1_id is not None and nomic_d2_name_stripped:
            nomic_d2_id = desc_to_id_lookup.get((2, nomic_d2_name_stripped))
            if nomic_d2_id is not None:
                claude_depth2 = claude_depth2_mapping.get((nomic_d1_id, nomic_d2_id))
                if not claude_depth2:
                    missing_nomic_topic_ids_depth2.add((nomic_d1_id, nomic_d2_id))
            elif nomic_d2_name_stripped:
                missing_nomic_topic_ids_depth2.add(f"NameNotInMeta:D2:{nomic_d1_name_stripped}->{nomic_d2_name_stripped}")

        if nomic_d1_id is not None and nomic_d2_id is not None and nomic_d3_name_stripped:
            nomic_d3_id = desc_to_id_lookup.get((3, nomic_d3_name_stripped))
            if nomic_d3_id is not None:
                claude_depth3 = claude_depth3_mapping.get((nomic_d1_id, nomic_d2_id, nomic_d3_id))
                if not claude_depth3:
                    missing_nomic_topic_ids_depth3.add((nomic_d1_id, nomic_d2_id, nomic_d3_id))
            elif nomic_d3_name_stripped:
                 missing_nomic_topic_ids_depth3.add(f"NameNotInMeta:D3:{nomic_d1_name_stripped}->{nomic_d2_name_stripped}->{nomic_d3_name_stripped}")
                
        comparison_data.append({
            'Datum ID': index,
            'Indexed Field': row.get(indexed_field),
            'NomicAtlasTopicDepth1': nomic_d1_name_raw, # Output raw names from topics.df
            'NomicAtlasTopicDepth2': nomic_d2_name_raw,
            'NomicAtlasTopicDepth3': nomic_d3_name_raw,
            'NomicClaudeTopicDepth1': claude_depth1,
            'NomicClaudeTopicDepth2': claude_depth2,
            'NomicClaudeTopicDepth3': claude_depth3,
        })
    
    if missing_nomic_topic_ids_depth1:
        print(f"WARNING: {len(missing_nomic_topic_ids_depth1)} unique Nomic Depth 1 topic IDs/Names are missing Claude mappings (e.g., {list(missing_nomic_topic_ids_depth1)[:3]})")
    if missing_nomic_topic_ids_depth2:
        print(f"WARNING: {len(missing_nomic_topic_ids_depth2)} unique Nomic Depth 2 topic ID paths/Names are missing Claude mappings (e.g., {list(missing_nomic_topic_ids_depth2)[:3]})")
    if missing_nomic_topic_ids_depth3:
        print(f"WARNING: {len(missing_nomic_topic_ids_depth3)} unique Nomic Depth 3 topic ID paths/Names are missing Claude mappings (e.g., {list(missing_nomic_topic_ids_depth3)[:3]})")
    
    return pd.DataFrame(comparison_data)

def get_topic_hierarchy_from_claude(anthropic_client, report, actual_max_depth):
    
    base_child_properties = {
        "child_id": {
            "type": "integer",
            "description": "The Nomic topic_id of the child topic, exactly as provided in the input report. This ID must be returned."
        },
        "child_name": {
            "type": "string",
            "description": "Short descriptive name for the child topic. SUPER IMPORTANT: MUST BE 5-20 CHARACTERS AND IS REQUIRED FOR EVERY CHILD."
        }
    }
    required_child_fields = ["child_id", "child_name"]

    if actual_max_depth == 3:
        base_child_properties["grandchild_topics"] = {
            "type": "array",
            "description": "Array of grandchild topics for this child. The number of grandchild topics for this child_id MUST EXACTLY MATCH the input. EACH CHILD MUST HAVE ALL ITS GRANDCHILDREN NAMED.",
            "items": {
                "type": "object",
                "properties": {
                    "grandchild_id": {
                        "type": "integer",
                        "description": "The Nomic topic_id of the grandchild topic, exactly as provided in the input report. This ID must be returned."
                    },
                    "grandchild_name": {
                        "type": "string",
                        "description": "Short descriptive name for the grandchild topic. SUPER IMPORTANT: MUST BE 5-20 CHARACTERS AND IS REQUIRED FOR EVERY GRANDCHILD."
                    }
                },
                "required": ["grandchild_id", "grandchild_name"]
            }
        }
        required_child_fields.append("grandchild_topics")

    parent_topics_description = "Array of parent topics at depth 1."
    if actual_max_depth == 2:
        parent_topics_description += " YOU MUST PROVIDE NAMES FOR ALL PARENTS (identified by parent_id) AND ALL THEIR CHILDREN (identified by child_id). The number of children for each parent_id MUST EXACTLY MATCH the input structure."
    else:  # actual_max_depth == 3
        parent_topics_description += " YOU MUST PROVIDE NAMES FOR ALL PARENTS (parent_id), ALL THEIR CHILDREN (child_id), AND ALL THEIR GRANDCHILDREN (grandchild_id). The number of children for each parent_id, and grandchildren for each child_id, MUST EXACTLY MATCH the input structure."

    topic_hierarchy_schema = {
        "name": "topic_hierarchy",
        "description": "Create a structured topic hierarchy with short description labels for EVERY node in the hierarchy, matching the exact input structure based on IDs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "parent_topics": {
                    "type": "array",
                    "description": parent_topics_description,
                    "items": {
                        "type": "object",
                        "properties": {
                            "parent_id": {
                                "type": "integer",
                                "description": "The Nomic topic_id of the parent topic, exactly as provided in the input report. This ID must be returned."
                            },
                            "parent_name": {
                                "type": "string",
                                "description": "Short descriptive name for the parent topic. SUPER IMPORTANT: MUST BE 5-20 CHARACTERS AND IS REQUIRED FOR EVERY PARENT."
                            },
                            "child_topics": {
                                "type": "array",
                                "description": "Array of child topics for this parent. The number of child topics for this parent_id MUST EXACTLY MATCH the input. EVERY PARENT MUST HAVE ALL ITS CHILDREN NAMED.",
                                "items": {
                                    "type": "object",
                                    "properties": base_child_properties,
                                    "required": required_child_fields
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

    system_prompt = '''You are a topological semiotician specializing in structured hierarchical ontology optimization.
Your task is to generate short, descriptive names (5-20 characters each) for topic nodes in a given hierarchy.
Each topic node in the input report is identified by a unique integer ID (e.g., `ID: 123`).

CRITICAL INSTRUCTIONS FOR STRUCTURAL INTEGRITY:
1.  Your output MUST use the exact `parent_id`, `child_id`, and `grandchild_id` values provided in the input report.
2.  For each `parent_id` from the input, the `child_topics` array in your output MUST contain objects for ALL and ONLY the `child_id`s that were listed under that parent in the input report. The number of children for a given `parent_id` MUST be identical to the input.
3.  If the input hierarchy has depth 3, for each `child_id` from the input, the `grandchild_topics` array in your output MUST contain objects for ALL and ONLY the `grandchild_id`s that were listed under that child in the input report. The number of grandchildren for a given `child_id` MUST be identical to the input.
4.  You MUST provide a descriptive name (e.g., `parent_name`, `child_name`, `grandchild_name`) for EVERY node ID present in the input structure.

Your primary goals for the descriptive names are:
1.  **Strict Hierarchical Validity**: Names must reflect the parent-child-grandchild relationships.
2.  **Maximal Conceptual Distinctness**: Differentiate concepts from siblings and other nodes.
3.  **Accuracy to Nomic Keywords**: Names must summarize the provided keywords for each node ID.
4.  **Clarity and Conciseness**: Names must be 5-20 characters. Use full words or standard abbreviations.

CRITICAL REMINDER: Adhere strictly to the input hierarchy structure using the provided IDs. Provide a name for EVERY ID. Do not add, omit, or re-parent any topics.
'''

    with anthropic_client.messages.stream(
        model=CLAUDE_MODEL_NAME,
        max_tokens=40000, 
        temperature=0.2,
        system=system_prompt,
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
    
    result = final_message.content[0].input
    
    print("\nValidating topic hierarchy...")
    is_valid = validate_topic_hierarchy_completeness(result, actual_max_depth)
    if not is_valid:
        print("WARNING: Some topics are missing names or IDs in Claude's response!")
    else:
        print("All topics have names and IDs. Claude response structure seems valid.")
    
    return result

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
    comparison_result_df = topic_comparison(data_df, topics_df, indexed_field, atlas_map, claude_depth1_mapping=depth1_mapping, claude_depth2_mapping=depth2_mapping, claude_depth3_mapping=depth3_mapping)            
    comparison_result_df.to_csv(f'{args.dataset}_topic_comparison.csv', index=False)
    