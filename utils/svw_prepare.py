#!/usr/bin/env python3
"""
Script to convert SV-Wild CSV dataset to standardized format.
Creates classes.txt, data.txt, train.txt, and test.txt files similar to UCF dataset format.
"""

import csv
import pathlib
from collections import defaultdict


def prepare_svw_dataset(csv_path: str, output_dir: str, split_idx: int = 1):
    """
    Convert SV-Wild CSV to standardized format.
    
    Args:
        csv_path: Path to SVW.csv file
        output_dir: Directory where output files will be created
        split_idx: Which train/test split to use (1, 2, or 3)
    """
    csv_path = pathlib.Path(csv_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV and collect data
    videos = []
    genres = set()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['FileName'].strip()
            genre = row['Genre'].strip()
            
            # Handle case sensitivity (Diving vs diving)
            genre_lower = genre.lower()
            if genre_lower == 'diving':
                genre = 'diving'  # Normalize to lowercase
            
            train_col = f'Train {split_idx}?'
            test_col = f'Test {split_idx}?'
            
            is_train = row.get(train_col, '').strip() == '1'
            is_test = row.get(test_col, '').strip() == '1'
            
            # Relative path: Videos/genre/filename
            rel_path = f"Videos/{genre}/{filename}"
            
            # Skip if genre is empty
            if not genre:
                continue
                
            videos.append({
                'path': rel_path,
                'genre': genre,
                'is_train': is_train,
                'is_test': is_test
            })
            genres.add(genre)
    
    # Sort genres for consistent ordering, filter out empty strings
    sorted_genres = sorted([g for g in genres if g])
    
    # Create genre to label mapping
    genre_to_label = {genre: idx for idx, genre in enumerate(sorted_genres)}
    
    # Write classes.txt
    classes_file = output_dir / 'classes.txt'
    with open(classes_file, 'w', encoding='utf-8') as f:
        for genre in sorted_genres:
            f.write(f"{genre}\n")
    print(f"Created {classes_file} with {len(sorted_genres)} classes")
    
    # Write data.txt (all videos)
    data_file = output_dir / 'data.txt'
    with open(data_file, 'w', encoding='utf-8') as f:
        for video in videos:
            label = genre_to_label[video['genre']]
            f.write(f"{video['path']} {label}\n")
    print(f"Created {data_file} with {len(videos)} videos")
    
    # Write train.txt
    train_videos = [v for v in videos if v['is_train']]
    train_file = output_dir / 'train.txt'
    with open(train_file, 'w', encoding='utf-8') as f:
        for video in train_videos:
            label = genre_to_label[video['genre']]
            f.write(f"{video['path']} {label}\n")
    print(f"Created {train_file} with {len(train_videos)} videos")
    
    # Write test.txt
    test_videos = [v for v in videos if v['is_test']]
    test_file = output_dir / 'test.txt'
    with open(test_file, 'w', encoding='utf-8') as f:
        for video in test_videos:
            label = genre_to_label[video['genre']]
            f.write(f"{video['path']} {label}\n")
    print(f"Created {test_file} with {len(test_videos)} videos")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total classes: {len(sorted_genres)}")
    print(f"  Total videos: {len(videos)}")
    print(f"  Train videos: {len(train_videos)}")
    print(f"  Test videos: {len(test_videos)}")
    print(f"  Using split {split_idx}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert SV-Wild CSV to standardized format"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="datasets/sv_wild/SVW.csv",
        help="Path to SVW.csv file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/sv_wild/sv_wild",
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Which train/test split to use (1, 2, or 3)"
    )
    
    args = parser.parse_args()
    prepare_svw_dataset(args.csv, args.output, args.split)

