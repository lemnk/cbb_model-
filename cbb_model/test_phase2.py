#!/usr/bin/env python3
import os
print("Testing Phase 2 file structure...")
required_files = ["src/features/__init__.py", "src/features/team_features.py", "src/features/player_features.py", "src/features/dynamic_features.py", "src/features/market_features.py", "src/features/feature_utils.py", "src/features/feature_pipeline.py", "PHASE2_SUMMARY.md", "notebooks/feature_exploration.py"]
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path}")
