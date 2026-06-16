import re
import os

files_to_fix = [
    'production/main.py',
    'production/execution/live/market_stream.py',
    'production/src/data/quality_gate.py',
    'production/src/alpha/prediction_registry.py',
    'production/dashboard/api/api_server.py'
]

for file_path in files_to_fix:
    full_path = os.path.join('/Users/pandu/Desktop/institutional-quant-research-os', file_path)
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        continue
        
    with open(full_path, 'r') as f:
        content = f.read()
        
    # Check if 'from datetime import timezone' is present
    if 'from datetime import timezone' not in content and 'import timezone' not in content:
        # Add it after 'from datetime import datetime' or similar
        content = re.sub(r'from datetime import (.*)', r'from datetime import \1, timezone', content, count=1)
        if 'timezone' not in content:
            # Fallback
            content = "from datetime import timezone\n" + content
            
    # Replace datetime.now() with datetime.now(timezone.utc)
    # Be careful not to replace datetime.now(timezone.utc) again
    content = re.sub(r'datetime\.now\(\)(?!\.replace)', r'datetime.now(timezone.utc)', content)
    # Also fix some cases where it might have been datetime.now().time() etc.
    
    with open(full_path, 'w') as f:
        f.write(content)
    print(f"Fixed {full_path}")

