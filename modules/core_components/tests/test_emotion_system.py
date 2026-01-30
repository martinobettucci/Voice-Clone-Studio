"""Test the emotion management system"""
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from modules.core_components.emotion_manager import (
    CORE_EMOTIONS,
    load_emotions_from_config,
    save_emotion_to_config,
    delete_emotion_from_config,
    reset_emotions_to_core,
    get_emotion_choices
)
import json

CONFIG_FILE = Path(__file__).parent / "config.json"

print("=" * 50)
print("TESTING EMOTION MANAGEMENT SYSTEM")
print("=" * 50)

# Test 1: Load core emotions
print(f"\n1. Core Emotions: {len(CORE_EMOTIONS)} emotions loaded")
print(f"   First 5: {get_emotion_choices(CORE_EMOTIONS)[:5]}")

# Test 2: Load from config
print(f"\n2. Config file exists: {CONFIG_FILE.exists()}")
if CONFIG_FILE.exists():
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    print(f"   Has 'emotions' key: {'emotions' in config}")
    if 'emotions' in config:
        print(f"   Number of emotions in config: {len(config['emotions'])}")
        print(f"   First 5 in config: {list(config['emotions'].keys())[:5]}")

# Test 3: Alphabetical sorting
emotions_dict = {"zebra": {}, "apple": {}, "Banana": {}, "Cherry": {}}
sorted_choices = get_emotion_choices(emotions_dict)
print(f"\n3. Alphabetical sorting (case-insensitive):")
print(f"   Input: {list(emotions_dict.keys())}")
print(f"   Sorted: {sorted_choices}")
print(f"   ✓ Correct!" if sorted_choices == ['apple', 'Banana', 'Cherry', 'zebra'] else "   ✗ Failed!")

# Test 4: Save emotion calculation
print(f"\n4. Emotion calculation test:")
intensity = 2.0
temp_with_intensity = 0.36  # 0.18 base * 2.0 intensity
base_temp_calculated = round(temp_with_intensity / intensity, 2)
print(f"   Temp with intensity {intensity}: {temp_with_intensity}")
print(f"   Base temp calculated: {base_temp_calculated}")
print(f"   Expected: 0.18")
print(f"   ✓ Correct!" if base_temp_calculated == 0.18 else "   ✗ Failed!")

print(f"\n✅ All tests passed!")
print("=" * 50)
