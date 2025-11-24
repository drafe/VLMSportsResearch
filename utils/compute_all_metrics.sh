#!/bin/bash

# Script to compute metrics for all result files
# Usage: ./utils/compute_all_metrics.sh

RESULTS_DIR="results"
OUTPUT_FILE="result_metrics.txt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Computing metrics for all result files..."
echo "Results directory: $RESULTS_DIR"
echo "Output file: $OUTPUT_FILE"
echo ""

# Clear/create output file
echo "Metrics computed on: $(date)" > "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Process each result file
total_files=0
processed=0
errors=0

for result_file in "$RESULTS_DIR"/*.txt; do
    if [ ! -f "$result_file" ]; then
        continue
    fi
    
    total_files=$((total_files + 1))
    filename=$(basename "$result_file")
    
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Processing: $filename${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    
    # Write to output file
    echo "========================================" >> "$OUTPUT_FILE"
    echo "File: $filename" >> "$OUTPUT_FILE"
    echo "========================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Extract dataset from filename
    if [[ "$filename" == svw_* ]]; then
        dataset_type="svw"
    elif [[ "$filename" == ucf_* ]]; then
        dataset_type="ucf"
    else
        dataset_type=""
    fi
    
    # Run compute_metrics and save to file, filtering out uv package installation messages
    if [ -n "$dataset_type" ]; then
        uv run python utils/compute_metrics.py "$result_file" --dataset "$dataset_type" 2>&1 | grep -v -E "^(Uninstalled|Installed) [0-9]+ package" >> "$OUTPUT_FILE"
        exit_code=${PIPESTATUS[0]}
    else
        uv run python utils/compute_metrics.py "$result_file" 2>&1 | grep -v -E "^(Uninstalled|Installed) [0-9]+ package" >> "$OUTPUT_FILE"
        exit_code=${PIPESTATUS[0]}
    fi
    
    if [ $exit_code -eq 0 ]; then
        processed=$((processed + 1))
        echo -e "${GREEN}✓ Successfully processed: $filename${NC}"
    else
        errors=$((errors + 1))
        echo -e "✗ Error processing: $filename"
    fi
    
    echo "" >> "$OUTPUT_FILE"
    echo ""
done

# Add summary to output file
echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "Summary:" >> "$OUTPUT_FILE"
echo "  Total files: $total_files" >> "$OUTPUT_FILE"
echo "  Processed: $processed" >> "$OUTPUT_FILE"
echo "  Errors: $errors" >> "$OUTPUT_FILE"

echo ""
echo "Summary:"
echo "  Total files: $total_files"
echo "  Processed: $processed"
echo "  Errors: $errors"
echo ""
echo -e "${GREEN}All results saved to: $OUTPUT_FILE${NC}"
