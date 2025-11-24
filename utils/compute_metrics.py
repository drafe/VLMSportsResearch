import argparse, re, sys
from pathlib import Path
from typing import Callable, Sequence, Tuple, List, Dict, Optional
from tqdm import tqdm

import numpy as np
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)


def get_class_names_for_dataset(dataset_type: Optional[str]) -> List[str]:
    """
    Get list of class names for a dataset.
    
    Args:
        dataset_type: 'svw' or 'ucf'
    
    Returns:
        List of class names in order (index corresponds to class number)
    """
    if dataset_type == 'svw':
        # SVW has 30 classes
        return [
            'archery', 'baseball', 'basketball', 'bmx', 'bowling', 'boxing',
            'cheerleading', 'discusthrow', 'diving', 'football', 'golf', 'gymnastics',
            'hammerthrow', 'highjump', 'hockey', 'hurdling', 'javelin', 'longjump',
            'polevault', 'rowing', 'running', 'shotput', 'skating', 'skiing',
            'soccer', 'swimming', 'tennis', 'volleyball', 'weight', 'wrestling'
        ]
    elif dataset_type == 'ucf':
        # UCF has 10 classes
        return [
            'diving', 'golfswing', 'kicking', 'lifting', 'ridinghorse',
            'running', 'skateboarding', 'swingbench', 'swingside', 'walking'
        ]
    else:
        return []


def normalize_ucf_class_name(class_name: str) -> str:
    """
    Normalize UCF class name from path to match class list.
    Handles variations like "Walk" -> "walking", "Run" -> "running".
    
    Args:
        class_name: Class name from path (e.g., "Walk", "Golf-Swing")
    
    Returns:
        Normalized class name matching the class list
    """
    normalized = normalize_class_name(class_name)
    
    # Map variations to correct class names
    ucf_mapping = {
        'walk': 'walking',
        'run': 'running',
        'golfswing': 'golfswing',  # already correct
        'golf': 'golfswing',  # "Golf-Swing" -> "golf" -> "golfswing"
        'diving': 'diving',
        'kicking': 'kicking',
        'lifting': 'lifting',
        'ridinghorse': 'ridinghorse',
        'riding': 'ridinghorse',  # "Riding-Horse" -> "riding" -> "ridinghorse"
        'skateboarding': 'skateboarding',
        'skateboard': 'skateboarding',
        'swingbench': 'swingbench',
        'swingside': 'swingside',
        'swing': 'swingside',  # Default: "Swing" usually means "Swing-Side"
    }
    
    return ucf_mapping.get(normalized, normalized)


def convert_number_to_class_name(prediction: str, dataset_type: Optional[str]) -> str:
    """
    Convert a numeric prediction to class name.
    
    Args:
        prediction: Prediction string (may be a number or class name)
        dataset_type: 'svw' or 'ucf'
    
    Returns:
        Class name if prediction is a number, otherwise returns prediction as-is
    """
    # Check if prediction is a number
    try:
        class_num = int(prediction.strip())
        class_names = get_class_names_for_dataset(dataset_type)
        if 0 <= class_num < len(class_names):
            return class_names[class_num]
    except (ValueError, TypeError):
        pass
    
    # If not a number, return as-is
    return prediction


def extract_last_word(text: str) -> str:
    """
    Extract the last word from the prediction text.
    
    Args:
        text: Full prediction text
    
    Returns:
        Last word in the text (cleaned of punctuation)
    """
    if not text:
        return ""
    
    # Split by whitespace and get all words
    words = re.findall(r'\b[A-Za-z-]+\b', text)
    
    if not words:
        return ""
    
    # Return the last word
    return words[-1]


def normalize_class_name(class_name: str) -> str:
    """
    Normalize class name: leave only letters, make lowercase.
    """
    return ''.join([c for c in class_name.lower() if c.isalpha()])

def parse_lines(raw_lines: Sequence[str], dataset_type: str) -> Tuple[List[int], List[int], str]:
    """
    Extract ground truth and predictions from tab-separated lines.
    
    Format: path\tprediction (prediction can span multiple lines)
    
    Ground truth is extracted from the path:
    - SVW: datasets/sv_wild/Videos/CLASS_NAME/... -> CLASS_NAME
    - UCF: datasets/ucf_sports_actions/ucf action/CLASS_NAME/... -> CLASS_NAME
    
    Prediction is the last word in the entire model output (for CoT prompts).
    
    Args:
        raw_lines: Lines from results file
        dataset_type: 'svw' or 'ucf' (auto-detected if None)
    
    Returns:
        Tuple of (ground_truth_classes, predicted_classes) as lists of strings
    """
    true, pred = [], []
    avg_time_per_video = ""  # Initialize as empty string
    
    # Pattern to match: path\tprediction (start of a new entry)
    pat = re.compile(r"(.+)\t(.+)\s*$")
    
    i = 0
    while i < len(raw_lines):
        ln = raw_lines[i].strip()
        
        # Skip empty lines and lines that don't match (like PROMPT: or time:)
        if not ln or ln.startswith("PROMPT:"):
            i += 1
            continue
        
        if ln.startswith("time:"):
            # Time is on the next non-empty line after "time:"
            i += 1
            # Skip empty lines
            while i < len(raw_lines) and not raw_lines[i].strip():
                i += 1
            # Get the time value from the next non-empty line
            if i < len(raw_lines):
                avg_time_per_video = raw_lines[i].strip()
            i += 1
            continue
        
        m = pat.match(ln)
        if not m:
            i += 1
            continue
        
        path, prediction_start = m.groups()
        
        # Collect the full prediction (may span multiple lines)
        prediction_lines = [prediction_start]
        i += 1
        
        # Continue collecting lines until we hit another path line or end of file
        while i < len(raw_lines):
            next_ln = raw_lines[i].strip()
            # Stop if we hit another path line (contains tab)
            if next_ln and '\t' in next_ln:
                break
            # Stop if we hit a special marker
            if next_ln.startswith("PROMPT:") or next_ln.startswith("time:"):
                break
            # Add all lines to prediction (including empty ones to preserve structure)
            prediction_lines.append(next_ln)
            i += 1
        
        # Join all prediction lines to get the full output
        full_prediction = '\n'.join(prediction_lines)
        
        # Extract ground truth from path
        path_parts = Path(path).parts
        
        
        # Extract class name from path
        if dataset_type == 'svw':
            # Format: .../Videos/CLASS_NAME/file.mp4
            if 'Videos' in path_parts:
                videos_idx = path_parts.index('Videos')
                if videos_idx + 1 < len(path_parts):
                    true_class = normalize_class_name(path_parts[videos_idx + 1])
                else:
                    continue
            else:
                continue
        elif dataset_type == 'ucf':
            # Format: .../ucf action/CLASS_NAME/...
            # Or: .../CLASS_NAME-Front/... or CLASS_NAME-Side/... or CLASS_NAME-SideAngle/...
            # Class names can have hyphens: "Golf-Swing", "Riding-Horse", "Swing-Bench", "Swing-Side"
            # Suffixes to remove: "-Front", "-Side", "-Back", "-SideAngle"
            if 'ucf action' in path:
                # Find the part after "ucf action"
                parts = path.split('ucf action')
                if len(parts) > 1:
                    remaining = parts[1].strip('/')
                    first_part = remaining.split('/')[0]
                    # Remove known suffixes (but preserve hyphens in class names like "Golf-Swing")
                    # Note: "Swing-SideAngle" should become "Swing-Side"
                    base_class = first_part
                    if base_class.endswith('-SideAngle'):
                        base_class = base_class[:-5]  # Replace "-SideAngle" with "-Side"
                    else:
                        suffixes = ['-Front', '-Side', '-Back']
                        for suffix in suffixes:
                            if base_class.endswith(suffix):
                                base_class = base_class[:-len(suffix)]
                                break
                    # Normalize UCF class name (handles "Walk" -> "walking", etc.)
                    true_class = normalize_ucf_class_name(base_class)
                else:
                    continue
            else:
                continue
        
        # Extract the last word from the full prediction
        prediction = extract_last_word(full_prediction)
        prediction = normalize_class_name(prediction)
        
        ds = get_class_names_for_dataset(dataset_type)
        if true_class not in ds:
            # Skip this entry if true class is not found
            continue
        
        true.append(ds.index(true_class))
        
        if prediction not in ds:
            # Invalid prediction -> assign to class n+1 (index = len(ds))
            pred.append(len(ds))
        else:
            pred.append(ds.index(prediction))
    
    return true, pred, avg_time_per_video


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute classification metrics from model results")
    ap.add_argument("file", nargs="?", help="Results file path (or read from stdin)")
    ap.add_argument("-d", "--dataset", choices=["svw", "ucf"], required=True, 
                    help="Dataset type (auto-detected if not specified)")
    args = ap.parse_args()

    # Read input
    raw = (Path(args.file).read_text(encoding="utf-8").splitlines()
           if args.file else
           sys.stdin.read().splitlines())

    # Parse lines to get class indices
    y_true, y_pred, avg_time_per_video = parse_lines(raw, args.dataset)
    
    if not y_true:
        print("Error: No valid predictions found in input file", file=sys.stderr)
        sys.exit(1)
    
    # Get number of classes (n classes + 1 for invalid predictions)
    ds = get_class_names_for_dataset(args.dataset)
    num_classes = len(ds) + 1  # n classes + 1 for invalid predictions
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    
    # For multiclass, use macro and weighted averages
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Prepare CSV data
    model_name = Path(args.file).stem if args.file else "STDIN"
    csv_line = f"{model_name},{acc:.4f},{prec_macro:.4f},{prec_weighted:.4f},{rec_macro:.4f},{rec_weighted:.4f},{f1_macro:.4f},{f1_weighted:.4f},{avg_time_per_video}"
    
    # Determine CSV file path (same directory as input file, or current directory)
    csv_path = Path("metrics.csv")
    
    # Check if file exists
    file_exists = csv_path.exists()
    
    # Write to CSV file
    with open(csv_path, 'a', encoding='utf-8') as f:
        if not file_exists:
            # Write header if file is new
            f.write("Model,Accuracy,Precision (macro),Precision (weighted),Recall (macro),Recall (weighted),F1 (macro),F1 (weighted),Avg. Time/video\n")
        f.write(csv_line + "\n")


if __name__ == "__main__":
    main()
