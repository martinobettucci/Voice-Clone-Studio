"""Test config initialization with emotions"""
import sys
from pathlib import Path
import json

# Simulate what voice_clone_studio.py does
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from modules.core_components.emotion_manager import CORE_EMOTIONS, load_emotions_from_config

CONFIG_FILE = Path(__file__).parent / "config.json"

# Load config (mimicking load_config function)
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
else:
    config = {}

print("Before initialization:")
print(f"  'emotions' in config: {'emotions' in config}")

# Initialize emotions if not present (what the app does)
if not config.get("emotions"):
    config["emotions"] = dict(sorted(CORE_EMOTIONS.items(), key=lambda x: x[0].lower()))
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print("\n✅ Emotions initialized in config.json!")

print(f"\nAfter initialization:")
print(f"  'emotions' in config: {'emotions' in config}")
print(f"  Number of emotions: {len(config.get('emotions', {}))}")
print(f"  First 5: {list(config['emotions'].keys())[:5]}")

# Test loading active emotions
_active_emotions = load_emotions_from_config(config)
print(f"\n✅ Active emotions loaded: {len(_active_emotions)}")
print(f"  First 5: {list(_active_emotions.keys())[:5]}")
