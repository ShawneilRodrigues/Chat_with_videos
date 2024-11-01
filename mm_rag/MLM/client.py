from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union, Iterator
import base64
import json
import requests
import subprocess
from utils import isBase64, encode_image_from_path_or_url

class BaseClient(ABC):
    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    @abstractmethod
    def generate(self, 
                 prompt: str,
                 image: str,
                 **kwargs
        ) -> str:
        """Send request to visual language model API and return generated text."""
        

class OllamaLLavaClient(BaseClient):
    def __init__(self, model: str = "llava", timeout: int = 60):
        super().__init__(timeout=timeout)
        self.model = model  # Specify the LLava model for `ollama`

    def generate(self, prompt: str, image: str, **kwargs) -> str:
        # Ensure image is base64 encoded
        if not isBase64(image):
            image = encode_image_from_path_or_url(image)
        
        # Convert the image and prompt to the JSON format required by `ollama`
        input_data = json.dumps({
            "prompt": prompt,
            "image": image
        })
        
        # Use `ollama` CLI to make the API call
        try:
            result = subprocess.run(
                ["ollama", "generate", self.model, input_data],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Check for errors
            if result.returncode != 0:
                raise RuntimeError(f"Ollama Error: {result.stderr}")
                
            # Parse and return response
            response_data = json.loads(result.stdout)
            return response_data.get("generated_text", "No response from model.")
        
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")

# Example usage:
client = OllamaLLavaClient(model="llava")
response = client.generate(prompt="Describe the scene", image="path_to_image_or_base64")
print("Generated response:", response)
