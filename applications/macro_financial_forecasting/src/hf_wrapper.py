from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel, ValidationError

class HuggingFaceChat:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    class Completions:
        def __init__(self, parent):
            self.parent = parent

        async def create(self, response_model: BaseModel, messages, max_retries=3):
            prompt = messages[0].get("content", "")
            result = await self.parent.client.text_generation(prompt, model=self.parent.model)
            generated_text = result[0]["generated_text"]

            # Validate and parse using Pydantic response_model if provided
            if response_model:
                try:
                    parsed = response_model.parse_raw(generated_text)
                    return parsed
                except ValidationError as e:
                    raise ValueError(f"Response validation failed: {e}") from e
            else:
                # Return raw text if no model provided
                return generated_text

    def __post_init__(self):
        self.chat = HuggingFaceChat.Completions(self)

class HFInstructorClient:
    def __init__(self, model="gpt2"):
        self.client = AsyncInferenceClient()
        self.model = model
        self.chat = HuggingFaceChat(self.client, self.model)
        self.chat.completions = HuggingFaceChat.Completions(self.chat)