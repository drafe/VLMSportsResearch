import torch, pathlib, yaml, argparse
from tqdm import tqdm
from loguru import logger
import time
import numpy as np
from scipy.stats import t
import os
import sys

# Add project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logger.add("errors.log", level="ERROR")

def mean_ci_halfwidth(values, alpha=0.05):

    x = np.asarray(values, dtype=float)
    mean = x.mean()
    se   = x.std(ddof=1) / np.sqrt(len(x))        
    h    = t.ppf(1 - alpha/2, df=len(x) - 1) * se 
    return mean, h

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("dataset_config")
    args = parser.parse_args()


    with open(args.config, 'r') as f: cfg = yaml.safe_load(f)
    with open(args.dataset_config, 'r') as f: dataset_cfg = yaml.safe_load(f)

    MODEL = cfg['model']
    MODEL_ID = f"{cfg['model_space']}/{MODEL}"
    CACHE_DIR = cfg['cache_dir']
    DATASET_ROOT = pathlib.Path(dataset_cfg['dataset_path'])
    # Use meta_test_path by default, can be overridden in model config
    meta_path_relative = cfg.get('meta_path') or dataset_cfg.get('meta_test_path')
    if not meta_path_relative:
        raise ValueError("meta_path must be specified either in model config or dataset config (meta_test_path)")
    # Resolve meta_path relative to dataset_root
    META_PATH = DATASET_ROOT / meta_path_relative
    # Dataset base path: use the parent directory of metadata file as base
    # This handles cases where metadata is in a subdirectory like "ucf action/"
    DATASET_PATH = META_PATH.parent
    OUTPUT_RESULTS = f"results/{MODEL}_{cfg['output_prefix']}.txt"
    FPS = cfg['fps']
    NUM_FRAMES = cfg['num_frames']
    MAX_NEW_TOKENS = cfg['max_new_tokens']

    with open(cfg['prompt'], 'r') as file: PROMPT = file.read().strip()
    logger.info(f"Model config: {cfg}")
    logger.info(f"Dataset config: {dataset_cfg}")
    logger.info(f"Prompt: {PROMPT}")

    CONTINUE_FROM=cfg['continue_from']

    if cfg['model_space']=="google":
        # Use GemmaAdapter for gemma-3 models, Gemma3nAdapter for gemma-3n models
        if "gemma-3n" in MODEL.lower() or "3n" in MODEL.lower():
            from models import Gemma3nAdapter as Model
            logger.info("gemma3n imported")
        else:
            from models import GemmaAdapter as Model
            logger.info("gemma3 imported")
    else: raise NotImplementedError

    if cfg.get("adapter_id"):
        backend = Model(model_id=MODEL_ID, cache_dir=CACHE_DIR, adapter_id=cfg['adapter_id'])
    else:
        backend = Model(model_id=MODEL_ID, cache_dir=CACHE_DIR)
    
    ds = UCFDataset(DATASET_PATH, META_PATH)
    logger.success("dataset initialised")

    if not CONTINUE_FROM:
        with open(OUTPUT_RESULTS, "w") as f_out:
            f_out.write(f"PROMPT:\n\n\n\n{PROMPT}\n\n\n\n")
    results = []
    with torch.inference_mode():
        for idx in tqdm(range(len(ds)), total=len(ds)):
            try:
                start = time.time()
                sample = ds[idx]
                if CONTINUE_FROM:
                    if idx<CONTINUE_FROM: continue
                inputs = backend.encode_query(sample["path"], PROMPT, fps=FPS, num_frames=NUM_FRAMES)
                reply  = backend.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
                end = time.time()

                line = f"{sample['path']}\t{reply}\n"
                with open(OUTPUT_RESULTS, "a") as f_out:
                    f_out.write(line)
                logger.info(line.strip())
                logger.info(str(end-start))
                results.append(end-start)

            except Exception as e:
                logger.error(sample['path'])
                logger.error(e)
                exit()


    m, hw = mean_ci_halfwidth(results)
    logger.info(f"{m:.4f} ±{hw:.4f}")
    with open(OUTPUT_RESULTS, "a") as f_out:
        f_out.write(f"time:\n\n\n\n{m:.4f} ±{hw:.4f}\n\n\n\n")
