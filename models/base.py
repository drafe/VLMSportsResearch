from abc import ABC, abstractmethod

class VLMBackend(ABC):
    """
    Minimal contract every vision-language model adapter must satisfy.
    """

    @abstractmethod
    def __init__(self, model_id: str, cache_dir: str, **kwargs): raise NotImplementedError
    
    @abstractmethod
    def encode_query(self, video_path: str, prompt: str): raise NotImplementedError
    
    @abstractmethod
    def generate(self, inputs, max_new_tokens: int) -> str: raise NotImplementedError
