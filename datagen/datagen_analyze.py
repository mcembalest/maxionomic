import json
from pathlib import Path
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List
import statistics
from datetime import datetime

def analyze_value(value: Any) -> dict:
    """Analyze a single value"""
    if isinstance(value, (int, float)):
        return {
            'type': type(value).__name__,
            'min': value,
            'max': value,
            'sum': value,
            'count': 1
        }
    elif isinstance(value, str):
        # Try to parse as date
        try:
            datetime.fromisoformat(value)
            return {'type': 'date', 'values': Counter([value]), 'count': 1}
        except ValueError:
            return {'type': 'string', 'values': Counter([value]), 'count': 1}
    elif isinstance(value, bool):
        return {'type': 'boolean', 'values': Counter([str(value)]), 'count': 1}
    elif isinstance(value, list):
        return {'type': 'list', 'length': len(value), 'count': 1}
    elif value is None:
        return {'type': 'null', 'count': 1}
    return {'type': 'unknown', 'count': 1}

def merge_stats(old_stats: dict, new_stats: dict) -> dict:
    """Merge two statistics dictionaries"""
    if old_stats['type'] != new_stats['type']:
        return {'type': 'mixed', 'count': old_stats['count'] + new_stats['count']}
    
    result = {'type': old_stats['type'], 'count': old_stats['count'] + new_stats['count']}
    
    if old_stats['type'] in ('integer', 'float'):
        result.update({
            'min': min(old_stats['min'], new_stats['min']),
            'max': max(old_stats['max'], new_stats['max']),
            'sum': old_stats['sum'] + new_stats['sum']
        })
    elif old_stats['type'] in ('string', 'date', 'boolean'):
        result['values'] = old_stats['values'] + new_stats['values']
    elif old_stats['type'] == 'list':
        result['length'] = (old_stats['length'] * old_stats['count'] + 
                          new_stats['length'] * new_stats['count']) / result['count']
    
    return result

def analyze_object(obj: Dict, stats: defaultdict = None) -> defaultdict:
    """Recursively analyze an object and update statistics"""
    if stats is None:
        stats = defaultdict(lambda: defaultdict(dict))
    
    for key, value in obj.items():
        # Analyze current value
        current_stats = analyze_value(value)
        
        # Update or merge stats
        if key not in stats:
            stats[key] = current_stats
        else:
            stats[key] = merge_stats(stats[key], current_stats)
        
        # Recurse into nested objects and lists
        if isinstance(value, dict):
            nested_stats = analyze_object(value)
            for nested_key, nested_value in nested_stats.items():
                full_key = f"{key}.{nested_key}"
                if full_key not in stats:
                    stats[full_key] = nested_value
                else:
                    stats[full_key] = merge_stats(stats[full_key], nested_value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    nested_stats = analyze_object(item)
                    for nested_key, nested_value in nested_stats.items():
                        full_key = f"{key}[].{nested_key}"
                        if full_key not in stats:
                            stats[full_key] = nested_value
                        else:
                            stats[full_key] = merge_stats(stats[full_key], nested_value)
    
    return stats

def format_stats(stats: dict, key: str = "", indent: int = 0) -> List[str]:
    """Format statistics for display"""
    lines = []
    padding = "  " * indent
    
    # Show field name
    if key:
        lines.append(f"{padding}{key}:")
    
    padding = "  " * (indent + 1)
    
    # Show type and count
    lines.append(f"{padding}Type: {stats['type']}")
    lines.append(f"{padding}Count: {stats['count']:,}")
    
    # Type-specific statistics
    if stats['type'] in ('integer', 'float'):
        lines.append(f"{padding}Min: {stats['min']:,}")
        lines.append(f"{padding}Max: {stats['max']:,}")
        lines.append(f"{padding}Average: {stats['sum'] / stats['count']:,.2f}")
    elif stats['type'] in ('string', 'date', 'boolean'):
        # Show top 5 most common values
        lines.append(f"{padding}Unique values: {len(stats['values']):,}")
        if len(stats['values']) <= 10:
            for value, count in stats['values'].most_common():
                percentage = (count / stats['count']) * 100
                lines.append(f"{padding}- {value}: {count:,} ({percentage:.1f}%)")
        else:
            for value, count in stats['values'].most_common(5):
                percentage = (count / stats['count']) * 100
                lines.append(f"{padding}- {value}: {count:,} ({percentage:.1f}%)")
            lines.append(f"{padding}... {len(stats['values']) - 5:,} more values")
    elif stats['type'] == 'list':
        lines.append(f"{padding}Average length: {stats['length']:.1f}")
    
    return lines

def analyze_json_files(directory: str):
    """Analyze all JSON files in directory"""
    path = Path(directory)
    if not path.exists():
        print(f"Directory '{directory}' not found")
        return
    
    for file_path in path.glob("*.json"):
        print(f"\n{'='*80}\nAnalyzing {file_path.name}:\n{'='*80}")
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            # Analyze all objects
            stats = None
            for obj in data:
                if stats is None:
                    stats = analyze_object(obj)
                else:
                    new_stats = analyze_object(obj)
                    for key, value in new_stats.items():
                        if key not in stats:
                            stats[key] = value
                        else:
                            stats[key] = merge_stats(stats[key], value)
            
            # Print results
            for key, value in stats.items():
                print("\n".join(format_stats(value, key)))
                print()
                
        except Exception as e:
            print(f"Error analyzing {file_path.name}: {e}")

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "generated_data"
    analyze_json_files(directory)