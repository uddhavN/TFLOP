import json
import torch

def validate_tflop_dataset(dataset_path):
    """Check for samples that might cause empty table_breakdown"""
    problematic_samples = []
    
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                
                # Check for empty or missing critical data
                issues = []
                
                if not data.get('dr_coord') or len(data['dr_coord']) == 0:
                    issues.append("Empty dr_coord")
                
                if not data.get('gold_coord') or len(data['gold_coord']) == 0:
                    issues.append("Empty gold_coord")
                
                if data.get('num_rows', 0) == 0:
                    issues.append("Zero rows")
                    
                if data.get('num_cols', 0) == 0:
                    issues.append("Zero columns")
                
                if issues:
                    problematic_samples.append({
                        'index': i,
                        'file_name': data.get('file_name', 'unknown'),
                        'issues': issues
                    })
                    
            except Exception as e:
                problematic_samples.append({
                    'index': i,
                    'file_name': 'unknown',
                    'issues': [f"JSON parsing error: {e}"]
                })
    
    return problematic_samples

# Run validation
problems = validate_tflop_dataset('TFLOP-dataset/meta_data/validation.jsonl')
print(f"Found {len(problems)} problematic samples:")
for problem in problems[:10]:  # Show first 10
    print(f"Sample {problem['index']} ({problem['file_name']}): {problem['issues']}")
