import re
import os

file_path = '/Users/pandu/Desktop/institutional-quant-research-os/production/src/execution/live/market_stream.py'

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        
    if 'from datetime import timezone' not in content and 'import timezone' not in content:
        content = re.sub(r'from datetime import (.*)', r'from datetime import \1, timezone', content, count=1)
        if 'timezone' not in content:
            content = "from datetime import timezone\n" + content
            
    content = re.sub(r'datetime\.now\(\)(?!\.replace)', r'datetime.now(timezone.utc)', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"Fixed {file_path}")
