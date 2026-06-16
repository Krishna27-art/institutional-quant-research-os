import os
import re

files_to_fix = [
    "production/main.py",
    "production/src/execution/live/market_stream.py",
    "production/src/data/quality_gate.py",
    "production/src/alpha/prediction_registry.py",
    "production/dashboard/api/api_server.py"
]

for file_path in files_to_fix:
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}, does not exist")
        continue
        
    with open(file_path, "r") as f:
        content = f.read()
        
    if "datetime.now()" not in content:
        print(f"No datetime.now() in {file_path}")
        continue
        
    # Replace datetime.now() with datetime.now(timezone.utc)
    content = content.replace("datetime.now()", "datetime.now(timezone.utc)")
    
    # Check if timezone is imported
    if "from datetime import timezone" not in content and "import timezone" not in content:
        # Try to add it near the datetime import
        if "from datetime import datetime" in content:
            content = content.replace("from datetime import datetime", "from datetime import datetime, timezone")
        elif "from datetime import datetime, timedelta" in content:
            content = content.replace("from datetime import datetime, timedelta", "from datetime import datetime, timedelta, timezone")
        else:
            # Just add it at the top
            content = "from datetime import timezone\n" + content
            
    with open(file_path, "w") as f:
        f.write(content)
    print(f"Fixed {file_path}")

