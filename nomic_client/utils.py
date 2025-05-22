import numpy as np

def normalize(vec):
    arr = np.array(vec)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm

def format_iso8601(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')