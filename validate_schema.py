import csv
import sys
import os

sys.path.insert(0, os.path.abspath('code'))
from main import (
    ALLOWED_CLAIM_STATUS,
    ALLOWED_ISSUE_TYPES,
    ALLOWED_RISK_FLAGS,
    ALLOWED_SEVERITY,
    OBJECT_PARTS
)

def validate_schema():
    with open('dataset/output.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    errors = []
    
    for i, row in enumerate(rows, start=2):
        uid = row.get('user_id', f'Row {i}')
        row_errors = []
        
        # Check claim_status
        if row['claim_status'] not in ALLOWED_CLAIM_STATUS:
            row_errors.append(f"Invalid claim_status: {row['claim_status']}")
            
        # Check issue_type
        if row['issue_type'] not in ALLOWED_ISSUE_TYPES:
            row_errors.append(f"Invalid issue_type: {row['issue_type']}")
            
        # Check severity
        if row['severity'] not in ALLOWED_SEVERITY:
            row_errors.append(f"Invalid severity: {row['severity']}")
            
        # Check object_part against the specific claim_object
        obj = row['claim_object']
        allowed_parts = OBJECT_PARTS.get(obj, {"unknown"})
        if row['object_part'] not in allowed_parts:
            row_errors.append(f"Invalid object_part '{row['object_part']}' for object '{obj}'")
            
        # Check risk_flags
        flags = [f.strip() for f in row['risk_flags'].split(';') if f.strip()]
        for flag in flags:
            if flag not in ALLOWED_RISK_FLAGS:
                row_errors.append(f"Invalid risk_flag: {flag}")
                
        if row_errors:
            errors.append({"user_id": uid, "errors": row_errors})
            
    print(f"Schema validation checked {len(rows)} rows.")
    if errors:
        print(f"Found schema errors in {len(errors)} rows:")
        for err in errors:
            print(f"  {err['user_id']}: {', '.join(err['errors'])}")
    else:
        print("All 44 rows adhere perfectly to the schema and allowed enums!")

if __name__ == "__main__":
    validate_schema()
