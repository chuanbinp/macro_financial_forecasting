from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel, ValidationError

class HuggingFaceChat:
    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.completions = HuggingFaceChat.Completions(self)

    class Completions:
        def __init__(self, parent):
            self.parent = parent

        async def create(self, response_model: BaseModel, messages, max_retries=3):
            # For Llama Instruct: use chat_completion
            result = await self.parent.client.chat_completion(
                model=self.parent.model,
                messages=messages
            )

            generated_text = result.choices[0].message["content"]

            # If a response_model is given, validate & parse it
            if response_model:
                try:
                    return response_model.parse_raw(generated_text)
                except ValidationError as e:
                    raise ValueError(f"Response validation failed: {e}") from e

            return generated_text


class HFInstructorClient:
    def __init__(self, model="meta-llama/Llama-3.1-8B-Instruct"):
        self.client = AsyncInferenceClient()
        self.model = model
        self.chat = HuggingFaceChat(self.client, self.model)