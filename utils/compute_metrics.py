import argparse, re, sys
from pathlib import Path
from typing import Callable, Sequence, Tuple, List, Dict, Optional
from tqdm import tqdm

import numpy as np
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


def normalize_class_name(name: str) -> str:
    """Normalize class name for comparison: lowercase, remove hyphens."""
    return name.lower().replace('-', '').replace('_', '').strip()


def parse_lines(raw_lines: Sequence[str], dataset_type: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Extract ground truth and predictions from tab-separated lines.
    
    Format: path\tprediction
    
    Ground truth is extracted from the path:
    - SVW: datasets/sv_wild/Videos/CLASS_NAME/... -> CLASS_NAME
    - UCF: datasets/ucf_sports_actions/ucf action/CLASS_NAME/... -> CLASS_NAME
    
    Args:
        raw_lines: Lines from results file
        dataset_type: 'svw' or 'ucf' (auto-detected if None)
    
    Returns:
        Tuple of (ground_truth_classes, predicted_classes) as lists of strings
    """
    true, pred = [], []
    
    # Pattern to match: path\tprediction
    pat = re.compile(r"(.+)\t(.+)\s*$")
    
    for ln in raw_lines:
        ln = ln.strip()
        # Skip empty lines and lines that don't match (like PROMPT: or time:)
        if not ln or ln.startswith("PROMPT:") or ln.startswith("time:"):
            continue
        
        m = pat.match(ln)
        if not m:
            continue
        
        path, prediction = m.groups()
        
        # Extract ground truth from path
        path_parts = Path(path).parts
        
        # Auto-detect dataset type if not specified
        if dataset_type is None:
            if 'sv_wild' in path or 'Videos' in path_parts:
                dataset_type = 'svw'
            elif 'ucf_sports_actions' in path or 'ucf action' in path:
                dataset_type = 'ucf'
        
        # Extract class name from path
        if dataset_type == 'svw':
            # Format: .../Videos/CLASS_NAME/file.mp4
            if 'Videos' in path_parts:
                videos_idx = path_parts.index('Videos')
                if videos_idx + 1 < len(path_parts):
                    true_class = path_parts[videos_idx + 1]
                else:
                    continue
            else:
                continue
        elif dataset_type == 'ucf':
            # Format: .../ucf action/CLASS_NAME/...
            # Or: .../CLASS_NAME-Front/... or CLASS_NAME-Side/...
            if 'ucf action' in path:
                # Find the part after "ucf action"
                parts = path.split('ucf action')
                if len(parts) > 1:
                    remaining = parts[1].strip('/')
                    first_part = remaining.split('/')[0]
                    # Remove suffixes like "-Front", "-Side", "-Back"
                    true_class = re.sub(r'-[A-Za-z]+$', '', first_part)
                else:
                    continue
            else:
                continue
        else:
            # Fallback: try to extract from any path structure
            # Look for common patterns
            if len(path_parts) >= 2:
                # Try second-to-last or last directory
                true_class = path_parts[-2] if len(path_parts) >= 2 else path_parts[-1]
            else:
                continue
        
        # Normalize prediction (remove extra whitespace, handle case variations)
        prediction = prediction.strip()
        
        true.append(true_class)
        pred.append(prediction)
    
    return true, pred


def create_class_mapping(true_classes: List[str], pred_classes: List[str]) -> Dict[str, int]:
    """
    Create a mapping from class names to integer labels.
    Uses normalized class names to handle variations.
    """
    all_classes = set()
    
    # Add all true classes
    for cls in true_classes:
        all_classes.add(normalize_class_name(cls))
    
    # Add all predicted classes
    for cls in pred_classes:
        all_classes.add(normalize_class_name(cls))
    
    # Sort for consistent ordering
    sorted_classes = sorted(all_classes)
    return {cls: idx for idx, cls in enumerate(sorted_classes)}


def convert_to_labels(class_names: List[str], class_mapping: Dict[str, int]) -> List[int]:
    """Convert class names to integer labels using normalized comparison."""
    labels = []
    for cls in class_names:
        normalized = normalize_class_name(cls)
        # Find matching class in mapping
        label = class_mapping.get(normalized, -1)
        if label == -1:
            # Try to find closest match
            for mapped_cls, mapped_label in class_mapping.items():
                if normalize_class_name(mapped_cls) == normalized:
                    label = mapped_label
                    break
        labels.append(label)
    return labels


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    metric: Callable[[Sequence[int], Sequence[int]], float],
    n: int = 1_000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(rng_seed)
    pairs = list(zip(y_true, y_pred))
    vals = []

    for _ in tqdm(range(n), desc="Bootstrap"):
        sample = resample(pairs, n_samples=len(pairs), random_state=rng)
        yt, yp = zip(*sample)
        try:
            v = metric(yt, yp)
            if not np.isnan(v):
                vals.append(v)
        except (ValueError, ZeroDivisionError):
            continue

    if not vals:
        return float("nan"), float("nan")

    lower = np.percentile(vals, 100 * alpha / 2)
    upper = np.percentile(vals, 100 * (1 - alpha / 2))
    return lower, upper


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute classification metrics from model results")
    ap.add_argument("file", nargs="?", help="Results file path (or read from stdin)")
    ap.add_argument("-n", "--bootstraps", type=int, default=1_000, help="Number of bootstrap samples")
    ap.add_argument("-d", "--dataset", choices=["svw", "ucf"], default=None, 
                    help="Dataset type (auto-detected if not specified)")
    ap.add_argument("--show-confusion", action="store_true", help="Show confusion matrix")
    ap.add_argument("--show-report", action="store_true", help="Show detailed classification report")
    args = ap.parse_args()

    # Read input
    raw = (Path(args.file).read_text(encoding="utf-8").splitlines()
           if args.file else
           sys.stdin.read().splitlines())

    # Parse lines to get class names
    true_classes, pred_classes = parse_lines(raw, args.dataset)
    
    if not true_classes:
        print("Error: No valid predictions found in input file", file=sys.stderr)
        sys.exit(1)
    
    # Create class mapping
    class_mapping = create_class_mapping(true_classes, pred_classes)
    num_classes = len(class_mapping)
    
    # Convert to integer labels
    y_true = convert_to_labels(true_classes, class_mapping)
    y_pred = convert_to_labels(pred_classes, class_mapping)
    
    # Check for unmapped predictions
    unmapped = sum(1 for label in y_pred if label == -1)
    if unmapped > 0:
        print(f"Warning: {unmapped} predictions could not be mapped to known classes", file=sys.stderr)
        # Filter out unmapped predictions
        valid_indices = [i for i, label in enumerate(y_pred) if label != -1]
        y_true = [y_true[i] for i in valid_indices]
        y_pred = [y_pred[i] for i in valid_indices]
        true_classes = [true_classes[i] for i in valid_indices]
        pred_classes = [pred_classes[i] for i in valid_indices]
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    
    # For multiclass, use macro and weighted averages
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print(f"Samples         : {len(y_true)}")
    print(f"Classes         : {num_classes}")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision (macro)    : {prec_macro:.4f}")
    print(f"Precision (weighted): {prec_weighted:.4f}")
    print(f"Recall (macro)       : {rec_macro:.4f}")
    print(f"Recall (weighted)    : {rec_weighted:.4f}")
    print(f"F1-score (macro)    : {f1_macro:.4f}")
    print(f"F1-score (weighted) : {f1_weighted:.4f}")
    
    # LaTeX format
    print(f"\nLaTeX: {acc:.4f} & {prec_macro:.4f} & {rec_macro:.4f} & {f1_macro:.4f}")
    
    if args.show_confusion:
        print("\nConfusion Matrix:")
        print(cm)
    
    if args.show_report:
        # Create label names for report
        label_names = sorted(class_mapping.keys())
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   labels=list(range(num_classes)),
                                   target_names=label_names,
                                   zero_division=0))


if __name__ == "__main__":
    main()
