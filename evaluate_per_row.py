import csv
import sys

def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    sample_claims = load_csv('dataset/sample_claims.csv')
    output_claims = load_csv('code/evaluation/sample_predictions_per_row.csv')
    
    sample_dict = {row['user_id']: row for row in sample_claims}
    output_dict = {row['user_id']: row for row in output_claims}
    
    fields_to_check = [
        "evidence_standard_met",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "valid_image",
        "severity"
    ]
    
    mistakes = []
    
    for user_id, gold_row in sample_dict.items():
        if user_id in output_dict:
            pred_row = output_dict[user_id]
            row_mistakes = {}
            for field in fields_to_check:
                gold_val = gold_row.get(field, "").strip().lower()
                pred_val = pred_row.get(field, "").strip().lower()
                
                if field == "risk_flags":
                    gold_flags = set(f.strip() for f in gold_val.split(';') if f.strip() and f.strip() != 'none')
                    pred_flags = set(f.strip() for f in pred_val.split(';') if f.strip() and f.strip() != 'none')
                    if gold_flags != pred_flags:
                        row_mistakes[field] = f"Expected: {gold_flags if gold_flags else {'none'}}, Got: {pred_flags if pred_flags else {'none'}}"
                else:
                    if gold_val != pred_val:
                        row_mistakes[field] = f"Expected: '{gold_val}', Got: '{pred_val}'"
                        
            if row_mistakes:
                mistakes.append({"user_id": user_id, "errors": row_mistakes})
                
    if not mistakes:
        print("No mistakes found!")
    else:
        print(f"Found mistakes in {len(mistakes)} out of {len(sample_dict)} evaluated rows.\n")
        for m in mistakes:
            print(f"User: {m['user_id']}")
            for k, v in m['errors'].items():
                print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
