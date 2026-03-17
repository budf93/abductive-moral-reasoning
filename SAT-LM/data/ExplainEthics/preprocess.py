import pandas as pd
import json
import random
import os

random.seed(42)

data_dir = r"C:\Tugas_Akhir\ARGOS_public_anon\SAT-LM\data\data"

relationships = [
    'violate_fairness', 'violate_care', 'violate_authority', 
    'violate_liberty', 'violate_sanctity', 'violate_loyalty'
]

def preprocess_csv(csv_path, json_path):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if 'gold_foundation' not in df.columns:
        print(f"Skipping {csv_path} because 'gold_foundation' column is missing.")
        return
        
    out_data = []
    
    for _, row in df.iterrows():
        gold = row['gold_foundation']
        if pd.isna(gold):
            continue
            
        if random.random() < 0.5:
            label = gold
            gt = "true"
        else:
            other_labels = [r for r in relationships if r != gold]
            label = random.choice(other_labels)
            gt = "false"
            
        item = {
            "id": str(row['q_id']),
            "context": str(row['question']),
            "explanation": str(row['explanation']) if not pd.isna(row['explanation']) else "",
            "label": str(label),
            "gt": gt,
            "gold_foundation": str(gold)
        }
        
        for col in df.columns:
            if col not in ['q_id', 'question', 'explanation', 'gold_foundation'] and not pd.isna(row[col]):
                item[col] = str(row[col])
                
        out_data.append(item)
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f)
        
    print(f"Processed {csv_path} -> {json_path}")

preprocess_csv(os.path.join(data_dir, 'simple_question.csv'), os.path.join(data_dir, 'simple_question.json'))
preprocess_csv(os.path.join(data_dir, 'hard_question.csv'), os.path.join(data_dir, 'hard_question.json'))
