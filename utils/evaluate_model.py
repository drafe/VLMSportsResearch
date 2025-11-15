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
    OUTPUT_PREFIX = cfg['output_prefix']
    DATASET_ROOT = pathlib.Path(dataset_cfg['dataset_path'])
    DATASET_PREFIX = dataset_cfg['dataset_prefix']
    # Use meta_test_path by default, can be overridden in model config
    meta_path_relative = cfg.get('meta_path') or dataset_cfg.get('meta_test_path')
    if not meta_path_relative:
        raise ValueError("meta_path must be specified either in model config or dataset config (meta_test_path)")
    # Resolve meta_path relative to dataset_root
    META_PATH = DATASET_ROOT / meta_path_relative
    # Dataset base path: use the parent directory of metadata file as base
    # This handles cases where metadata is in a subdirectory like "ucf action/"
    DATASET_PATH = META_PATH.parent
    OUTPUT_RESULTS = f"results/{DATASET_PREFIX}_{MODEL}_{OUTPUT_PREFIX}.txt"
    FPS = cfg['fps']
    NUM_FRAMES = cfg['num_frames']
    MAX_NEW_TOKENS = cfg['max_new_tokens']

    # Determine prompt path: use prompt_type if specified, otherwise use prompt path
    if 'prompt_type' in cfg:
        prompt_type = cfg['prompt_type']
        # For SVW dataset, use svw_ prefix; for UCF, use no prefix
        if DATASET_PREFIX == "svw":
            prompt_path = f"prompts/svw_{prompt_type}.txt"
        else:
            prompt_path = f"prompts/{prompt_type}.txt"
    elif 'prompt' in cfg:
        prompt_path = cfg['prompt']
    else:
        raise ValueError("Either 'prompt_type' or 'prompt' must be specified in config")
    
    with open(prompt_path, 'r') as file: PROMPT = file.read().strip()
    logger.info(f"Model config: {cfg}")
    logger.info(f"Dataset config: {dataset_cfg}")
    logger.info(f"Prompt path: {prompt_path}")
    logger.info(f"Prompt preview: {PROMPT[:200]}...")

    CONTINUE_FROM=cfg['continue_from']

    if cfg['model_space']=="google":
        from models import GemmaAdapter as Model
        logger.info("gemma3 imported")
    elif cfg['model_space']=="Qwen":
        from models import Qwen25Adapter as Model
        logger.info("qwen2.5 imported")
    elif cfg['model_space']=="OpenGVLab":
        from models import InternVL3Adapter as Model
        logger.info("internvl3 imported")
    elif cfg['model_space']=="LanguageBind":
        from models import VideoLlavaAdapter as Model
        logger.info("llava imported")
    else: raise NotImplementedError

    if cfg.get("adapter_id"):
        backend = Model(model_id=MODEL_ID, cache_dir=CACHE_DIR, adapter_id=cfg['adapter_id'])
    else:
        backend = Model(model_id=MODEL_ID, cache_dir=CACHE_DIR)
    
    if DATASET_PREFIX == "ucf":
        from utils.datasets import UCFDataset
        ds = UCFDataset(DATASET_PATH, META_PATH)
    elif DATASET_PREFIX == "svw":
        from utils.datasets import SVWildDataset
        ds = SVWildDataset(DATASET_PATH, META_PATH)
    else:
        raise ValueError(f"Invalid dataset prefix: {DATASET_PREFIX}")
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

            except KeyboardInterrupt:
                exit(0)
            except BaseException as e:
                logger.error(f"{sample['path']}: {e}")


    m, hw = mean_ci_halfwidth(results)
    logger.info(f"{m:.4f} ±{hw:.4f}")
    with open(OUTPUT_RESULTS, "a") as f_out:
        f_out.write(f"time:\n\n\n\n{m:.4f} ±{hw:.4f}\n\n\n\n")
