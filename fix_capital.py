import os
import re

base_dir = '/Users/pandu/Desktop/institutional-quant-research-os/production'

# 1. Update settings.py
settings_path = os.path.join(base_dir, 'core/config/settings.py')
with open(settings_path, 'r') as f:
    settings_content = f.read()

if 'os.environ.get' not in settings_content:
    if 'import os' not in settings_content:
        settings_content = "import os\n" + settings_content
    settings_content = re.sub(r'TRADING_CAPITAL\s*=\s*250_000_000\.0', r"TRADING_CAPITAL = float(os.environ.get('TRADING_CAPITAL', 250_000_000.0))", settings_content)
    with open(settings_path, 'w') as f:
        f.write(settings_content)

# 2. Update other files
files_to_update = [
    'dashboard/api/api_server.py',
    'market_data/microstructure/liquidity.py',
    'src/execution/adapters/live_adapter.py'
]

import_statement = "from core.config.settings import TRADING_CAPITAL\n"

for relative_path in files_to_update:
    file_path = os.path.join(base_dir, relative_path)
    if not os.path.exists(file_path):
        continue
        
    with open(file_path, 'r') as f:
        content = f.read()
        
    if 'TRADING_CAPITAL' not in content:
        # Add import after other imports
        content = re.sub(r'(import .*?\n)', r'\1' + import_statement, content, count=1)
        
    # Replace the hardcoded numbers
    content = re.sub(r'250_000_000\.0', 'TRADING_CAPITAL', content)
    content = re.sub(r'250_000_000(?!_)', 'TRADING_CAPITAL', content)
    
    with open(file_path, 'w') as f:
        f.write(content)

print("Capital config updated.")
