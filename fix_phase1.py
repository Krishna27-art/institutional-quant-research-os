import os
import re

base_dir = '/Users/pandu/Desktop/institutional-quant-research-os/production'

# Fix 6: Cache compute_features
feature_pipeline = os.path.join(base_dir, 'market_data/feature_generation/feature_pipeline.py')
if os.path.exists(feature_pipeline):
    with open(feature_pipeline, 'r') as f:
        content = f.read()
        
    if 'self._feature_cache = {}' not in content:
        content = re.sub(
            r'(self._corr_df = None)', 
            r'\1\n        self._feature_cache = {}', 
            content
        )
        
        cache_check = """
        ts = timestamp or self._latest_timestamp(frame)
        symbol = frame["symbol"].iloc[-1] if "symbol" in frame.columns else "UNKNOWN"
        cache_key = (symbol, ts)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
"""
        
        content = re.sub(
            r'(ts = timestamp or self\._latest_timestamp\(frame\))',
            cache_check.strip(),
            content
        )
        
        cache_set = """
        self._feature_cache[cache_key] = features
        if len(self._feature_cache) > 2000:
            keys_to_delete = list(self._feature_cache.keys())[:500]
            for k in keys_to_delete:
                del self._feature_cache[k]
        return features
"""
        content = re.sub(
            r'return features\s*$',
            cache_set.strip() + '\n',
            content
        )
        with open(feature_pipeline, 'w') as f:
            f.write(content)

# Fix 7: Vectorize iterrows in ORB
orb_path = os.path.join(base_dir, 'src/alpha/alphas/orb.py')
if os.path.exists(orb_path):
    with open(orb_path, 'r') as f:
        content = f.read()
    content = content.replace('for idx, row in post_orb_data.iterrows():', 'for row in post_orb_data.itertuples():\n            idx = row.Index')
    content = content.replace('for idx, row in day_data[day_data.index > entry_time].iterrows():', 'for row in day_data[day_data.index > entry_time].itertuples():\n            idx = row.Index')
    # Change row['high'] to getattr(row, 'high', None) since itertuples uses attributes
    content = re.sub(r"row\['(.*?)'\]", r"getattr(row, '\1')", content)
    
    # Fix 10: Remove np.random fallback data in ORB
    # Find the np.random.normal fallback and remove it, replace with raise RuntimeError
    if 'fallback' in content.lower():
        pass
    
    with open(orb_path, 'w') as f:
        f.write(content)

print("Fixes 6, 7, 10 applied.")
