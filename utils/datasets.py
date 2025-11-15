import torch, pathlib


class UCFDataset(torch.utils.data.Dataset):
    def __init__(self, root: pathlib.Path, metapath):
        """
        UCF Sports Actions Dataset loader.
        
        Args:
            root: Path to dataset base directory (parent of metadata file, e.g., datasets/ucf_sports_actions/ucf action/)
            metapath: Path to metadata file (e.g., ucf action/data.txt or ucf action/train.txt)
                     Format: each line is "relative_path/to/video.avi label"
                     Paths in metadata are relative to root directory
        """
        self.root = pathlib.Path(root)
        self.video_paths = []
        self.labels = []
        
        with open(metapath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                # Format: "Golf-Swing-Front/004/RF1-13206_70024.avi 1"
                parts = line.rsplit(' ', 1)  # Split from right to handle paths with spaces
                if len(parts) == 2:
                    rel_path, label = parts
                    # Resolve path: root should contain "ucf action" subdirectory
                    video_path = self.root / rel_path
                    self.video_paths.append(video_path)
                    self.labels.append(int(label))
                else:
                    # Fallback: if no label, assume path only
                    video_path = self.root / line
                    self.video_paths.append(video_path)
                    self.labels.append(0)  # Default label

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = str(self.video_paths[idx])
        label = self.labels[idx]
        return {"path": path, "label": label}


class SVWildDataset(torch.utils.data.Dataset):
    def __init__(self, root: pathlib.Path, metapath):
        """
        SV-Wild Dataset loader.
        
        Args:
            root: Path to dataset base directory (parent of metadata file, e.g., datasets/sv_wild/)
            metapath: Path to metadata file (e.g., data.txt or train.txt)
                     Format: each line is "relative_path/to/video.avi label"
                     Paths in metadata are relative to root directory
        """
        self.root = pathlib.Path(root)
        self.video_paths = []
        self.labels = []
        
        with open(metapath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                # Format: "relative_path/to/video.avi label"
                parts = line.rsplit(' ', 1)  # Split from right to handle paths with spaces
                if len(parts) == 2:
                    rel_path, label = parts
                    # Resolve path relative to root
                    video_path = self.root / rel_path
                    self.video_paths.append(video_path)
                    self.labels.append(int(label))
                else:
                    # Fallback: if no label, assume path only
                    video_path = self.root / line
                    self.video_paths.append(video_path)
                    self.labels.append(0)  # Default label

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = str(self.video_paths[idx])
        label = self.labels[idx]
        return {"path": path, "label": label}

