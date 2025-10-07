#!/usr/bin/env python3
# filepath: /workspace/src/autodrive_f1tenth/autodrive_f1tenth/examine_model.py

import zipfile
import io
import torch
import os

def examine_model_zip(path):
    """Examine the contents of a model zip file"""
    if not path.endswith('.zip'):
        path += '.zip'
    
    print(f"Examining model at {path}...")
    
    with zipfile.ZipFile(path, 'r') as z:
        # List all files in the zip
        print("Files in zip archive:")
        for filename in z.namelist():
            print(f" - {filename}")
        
        # Look for policy.pth specifically
        if 'policy.pth' in z.namelist():
            print("\nExamining policy.pth...")
            with z.open('policy.pth') as f:
                buffer = io.BytesIO(f.read())
                policy_data = torch.load(buffer)
                
                print(f"Type of loaded data: {type(policy_data)}")
                
                # If it's a dict, print the keys
                if isinstance(policy_data, dict):
                    print("Dictionary keys:")
                    for key in policy_data.keys():
                        print(f" - {key}")

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "model.zip")
    examine_model_zip(model_path)