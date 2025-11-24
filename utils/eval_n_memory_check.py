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

    # Memory measurement functions
    def get_memory_stats():
        """Get current GPU memory statistics in MB"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
            max_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
            return {
                'allocated': allocated,
                'reserved': reserved,
                'max_allocated': max_allocated,
                'max_reserved': max_reserved
            }
        else:
            return {
                'allocated': 0.0,
                'reserved': 0.0,
                'max_allocated': 0.0,
                'max_reserved': 0.0
            }

    def reset_memory_stats():
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # Limit to 10 examples for quick testing
    MAX_EXAMPLES = 10
    num_examples = min(MAX_EXAMPLES, len(ds))
    logger.info(f"Running on {num_examples} examples (limited from {len(ds)})")

    if not CONTINUE_FROM:
        with open(OUTPUT_RESULTS, "w") as f_out:
            f_out.write(f"PROMPT:\n\n\n\n{PROMPT}\n\n\n\n")
            f_out.write("="*80 + "\n")
            f_out.write("Memory profiling enabled\n")
            f_out.write(f"Format: path\treply\ttime(s)\t"
                       f"mem_alloc_before(MB)\tmem_res_before(MB)\t"
                       f"mem_alloc_after(MB)\tmem_res_after(MB)\t"
                       f"mem_peak_alloc(MB)\tmem_peak_res(MB)\t"
                       f"mem_delta_alloc(MB)\tmem_delta_res(MB)\n")
            f_out.write("="*80 + "\n\n")

    # Get baseline memory (after model loading)
    baseline_memory = get_memory_stats()
    logger.info(f"Baseline memory after model load: {baseline_memory}")

    results = []
    memory_results = {
        'allocated_before': [],
        'reserved_before': [],
        'allocated_after': [],
        'reserved_after': [],
        'peak_allocated': [],
        'peak_reserved': [],
        'memory_delta_allocated': [],
        'memory_delta_reserved': []
    }

    with torch.inference_mode():
        for idx in tqdm(range(num_examples), total=num_examples):
            sample_path = None
            try:
                if CONTINUE_FROM:
                    if idx < CONTINUE_FROM:
                        continue
                
                # Reset peak memory stats before each request
                reset_memory_stats()
                
                # Measure memory before request
                memory_before = get_memory_stats()
                
                start = time.time()
                sample = ds[idx]
                sample_path = sample.get('path', f'index_{idx}')
                
                # Encode query
                inputs = backend.encode_query(sample["path"], PROMPT, fps=FPS, num_frames=NUM_FRAMES)
                
                # Generate response
                reply = backend.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
                end = time.time()
                
                # Measure memory after request
                memory_after = get_memory_stats()
                
                # Calculate deltas
                delta_allocated = memory_after['allocated'] - memory_before['allocated']
                delta_reserved = memory_after['reserved'] - memory_before['reserved']
                
                elapsed_time = end - start
                results.append(elapsed_time)
                
                # Store memory results
                memory_results['allocated_before'].append(memory_before['allocated'])
                memory_results['reserved_before'].append(memory_before['reserved'])
                memory_results['allocated_after'].append(memory_after['allocated'])
                memory_results['reserved_after'].append(memory_after['reserved'])
                memory_results['peak_allocated'].append(memory_after['max_allocated'])
                memory_results['peak_reserved'].append(memory_after['max_reserved'])
                memory_results['memory_delta_allocated'].append(delta_allocated)
                memory_results['memory_delta_reserved'].append(delta_reserved)
                
                # Write to file: path, reply, time, and all memory metrics
                line = (f"{sample_path}\t{reply}\t{elapsed_time:.4f}\t"
                       f"{memory_before['allocated']:.2f}\t"
                       f"{memory_before['reserved']:.2f}\t"
                       f"{memory_after['allocated']:.2f}\t"
                       f"{memory_after['reserved']:.2f}\t"
                       f"{memory_after['max_allocated']:.2f}\t"
                       f"{memory_after['max_reserved']:.2f}\t"
                       f"{delta_allocated:.2f}\t"
                       f"{delta_reserved:.2f}\n")
                
                with open(OUTPUT_RESULTS, "a") as f_out:
                    f_out.write(line)
                
                logger.info(f"{sample_path}: "
                          f"time: {elapsed_time:.4f}s, "
                          f"mem: {memory_before['allocated']:.2f} -> {memory_after['allocated']:.2f} MB, "
                          f"peak: {memory_after['max_allocated']:.2f} MB")

            except KeyboardInterrupt:
                exit(0)
            except BaseException as e:
                if sample_path is None:
                    sample_path = f'index_{idx}'
                logger.error(f"{sample_path}: {e}")
                # Write error to file
                with open(OUTPUT_RESULTS, "a") as f_out:
                    f_out.write(f"{sample_path}\tERROR: {e}\n")

    # Calculate statistics
    logger.info("\n" + "="*80)
    logger.info("Statistics Summary:")
    logger.info("="*80)
    
    with open(OUTPUT_RESULTS, "a") as f_out:
        f_out.write("\n" + "="*80 + "\n")
        f_out.write("Statistics Summary:\n")
        f_out.write("="*80 + "\n\n")
        
        # Time statistics
        if len(results) > 0:
            m, hw = mean_ci_halfwidth(results)
            logger.info(f"Time: {m:.4f} ± {hw:.4f} s")
            f_out.write(f"Time: {m:.4f} ± {hw:.4f} s\n\n")
        
        # Memory statistics
        f_out.write("Memory Statistics:\n")
        for metric_name, values in memory_results.items():
            if len(values) > 0:
                m, hw = mean_ci_halfwidth(values)
                logger.info(f"{metric_name}: {m:.2f} ± {hw:.2f} MB")
                f_out.write(f"{metric_name}: {m:.2f} ± {hw:.2f} MB\n")
        
        f_out.write("\nBaseline memory (after model load):\n")
        f_out.write(f"  Allocated: {baseline_memory['allocated']:.2f} MB\n")
        f_out.write(f"  Reserved: {baseline_memory['reserved']:.2f} MB\n")
    
    logger.info(f"\nBaseline memory (after model load): {baseline_memory}")
    logger.info(f"Results saved to: {OUTPUT_RESULTS}")
