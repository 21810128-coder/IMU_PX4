"""
================================================================================
RUN FULL PIPELINE - Entry Point
================================================================================
This script runs the complete corrected window feature extraction pipeline.

Author: Research Team
Date: 2024
================================================================================
"""

import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from window_feature_extractor import run_full_pipeline


if __name__ == "__main__":
    print("=" * 70)
    print("CORRECTED WINDOW FEATURE EXTRACTION")
    print("=" * 70)
    print("\nThis pipeline uses a band-pass filter (5-50 Hz) to properly")
    print("extract vibration from IMU data, avoiding the physically")
    print("implausible values obtained with high-pass filtering alone.")
    print("=" * 70)
    
    results = run_full_pipeline()
    
    print("\n" + "=" * 70)
    print("✅ ALL PROCESSING COMPLETED")
    print("=" * 70)