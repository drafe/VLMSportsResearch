from .base import VLMBackend

from loguru import logger

import torch
import numpy as np
import av
from huggingface_hub import hf_hub_download
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

class VideoLlavaAdapter(VLMBackend):
    def __init__(self, model_id: str, cache_dir: str, **kwargs):
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto",torch_dtype=torch.bfloat16, cache_dir=cache_dir).eval()
        logger.success("model loaded")

        self.processor = VideoLlavaProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        logger.success("processor inititalised")

    def encode_query(self, video_path: str, prompt: str, num_frames: int = 8, **kwargs):
        prompt = f"USER: <video>{prompt} ASSISTANT:"
        container = av.open(video_path)

        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
        clip = read_video_pyav(container, indices)

        inputs = self.processor(text=prompt, videos=clip, return_tensors="pt").to(self.model.device)
        return inputs
        

    def generate(self, inputs, max_new_tokens: int) -> str:
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # return self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        return self.processor.batch_decode(
            generate_ids[:, inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )[0]
