from .base import VLMBackend

from loguru import logger

from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Gemma3nForConditionalGeneration
from PIL import Image
import cv2
import torch, os, tempfile
from pathlib import Path

token = os.getenv("HF_TOKEN=")

def extract_frames(video_path: str, num_frames: int):
    """
    The function is adapted from:
    https://github.com/merveenoyan/smol-vision/blob/main/Gemma_3_for_Video_Understanding.ipynb
    """
    video_path = video_path if video_path.endswith("mp4") else f"{video_path}.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        assert False
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the step size to evenly distribute frames across the video.
    step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = round(frame_idx / fps, 2)
        frames.append((img, timestamp))

    cap.release()
    return frames

class GemmaAdapter(VLMBackend):
    def __init__(self, model_id: str, cache_dir: str, **kwargs):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=cache_dir, token=token
        ).eval()

        logger.success("model loaded")

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        logger.success("processor inititalised")

    def encode_query(self, video_path: str, prompt: str, num_frames: int = 8, **kwargs):
        video_frames = extract_frames(video_path, num_frames=num_frames)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        temp_dir = tempfile.TemporaryDirectory()
        for frame_data in video_frames:
            img, timestamp = frame_data
            messages[1]["content"].append({"type": "text", "text": f"Frame at {timestamp} seconds:"})
            img.save(f"{Path(temp_dir.name)}/frame_{timestamp}.png")
            messages[1]["content"].append({"type": "image", "url": f"{Path(temp_dir.name)}/frame_{timestamp}.png"})

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        return inputs



    def generate(self, inputs, max_new_tokens: int) -> str:
        input_length = inputs["input_ids"].shape[-1]
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        output = output[0][input_length:]
        response = self.processor.decode(output, skip_special_tokens=True)

        return response



class Gemma3nAdapter(GemmaAdapter):
    def __init__(self, model_id: str, cache_dir: str, **kwargs):
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=cache_dir, token=token
        ).eval()

        logger.success("model loaded")

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        logger.success("processor inititalised")