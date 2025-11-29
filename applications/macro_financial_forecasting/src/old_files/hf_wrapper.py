# import json
# import re
# from huggingface_hub import AsyncInferenceClient
# from pydantic import BaseModel, ValidationError


# class HFInstructorChat:
#     """Instructor-style wrapper for Hugging Face AsyncInferenceClient."""

#     def __init__(self, client: AsyncInferenceClient, model: str):
#         self.client = client
#         self.model = model
#         self.completions = HFInstructorCompletions(self)


# class HFInstructorCompletions:
#     def __init__(self, parent: HFInstructorChat):
#         self.parent = parent

#     async def create(
#         self,
#         response_model: type[BaseModel] | None,
#         messages: list[dict],
#         max_retries: int = 3
#     ):
#         """Main entry point, similar to instructor.Client.chat.completions.create."""

#         # ---- 1. Inject JSON enforcement if response_model is provided ----
#         if response_model:
#             messages = self._inject_schema_prompt(messages, response_model)

#         # ---- 2. Retry loop to enforce JSON validity ----
#         last_exception = None

#         for attempt in range(max_retries):
#             try:
#                 raw = await self._call_hf(messages)
#                 cleaned = self._clean_json_string(raw)

#                 if not response_model:
#                     return cleaned  # No schema requested → return raw text

#                 return response_model.parse_raw(cleaned)

#             except ValidationError as e:
#                 last_exception = e
#                 # try again after cleaning
#             except json.JSONDecodeError as e:
#                 last_exception = e
#                 # try again

#         # After retries exhausted
#         raise ValueError(f"Response validation failed after {max_retries} retries: {last_exception}")

#     async def _call_hf(self, messages: list[dict]) -> str:
#         """Low-level HF chat completion call."""
#         prompt = "\n".join([msg["content"] for msg in messages])
#         result = await self.parent.client.text_generation(
#             inputs=prompt,
#             model=self.parent.model
#         )
#         return result.choices[0].message["content"]

#     # ---------- JSON Prompt Injection ----------
#     def _inject_schema_prompt(self, messages, response_model):
#         """Inject Instructor-style JSON schema instructions."""
#         schema = response_model.schema_json(indent=2)

#         system_msg = {
#             "role": "system",
#             "content": (
#                 "You are a strict JSON generator.\n"
#                 "Return ONLY valid JSON that can be parsed without errors.\n"
#                 "Follow this exact JSON schema:\n"
#                 f"{schema}\n"
#                 "Do NOT include markdown, code fences, comments, or any text outside the JSON object."
#             )
#         }

#         # Prepend schema instructions
#         return [system_msg] + messages

#     # ---------- JSON Cleaning / Repair Logic ----------
#     def _clean_json_string(self, text: str) -> str:
#         """Fix common JSON formatting problems."""

#         # Remove markdown fences
#         text = re.sub(r"```json|```", "", text).strip()

#         # Attempt naive JSON load — if this works, return immediately
#         try:
#             json.loads(text)
#             return text
#         except Exception:
#             pass

#         # Try extracting first {...} block
#         m = re.search(r"\{[\s\S]*\}", text)
#         if m:
#             candidate = m.group(0)
#             try:
#                 json.loads(candidate)
#                 return candidate
#             except Exception:
#                 pass

#         # Try adding missing quotes around keys
#         candidate = re.sub(r"(\w+):", r'"\1":', text)
#         try:
#             json.loads(candidate)
#             return candidate
#         except Exception:
#             pass

#         # Last resort — return raw text and let Pydantic raise
#         return text


# # ---------- Final User-Facing Client ----------
# class HFInstructorClient:
#     def __init__(self, model: str):
#         self.client = AsyncInferenceClient()
#         self.model = model
#         self.chat = HFInstructorChat(self.client, self.model)
