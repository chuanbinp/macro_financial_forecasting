from typing import List
from agentics import AG  # as per notebook
from pydantic import BaseModel
from crewai import LLM  # example LLM wrapper
from config import Config


class AgenticTransducer:
    """Encapsulates agentic transduction logic for async LLM calls."""

    def __init__(self, config: Config, verbose: bool = False):
        self.llm = LLM(model=config.llm_model)
        self.verbose = verbose

    def create_AG(self, pydantic_class: BaseModel, data: List[BaseModel]) -> AG:
        return AG(atype=pydantic_class, states=data, llm=self.llm, verbose=self.verbose)

    async def self_transduce(self, ag_obj: AG, prompt: str) -> AG:
        """Run transduction on a single agentic object."""
        ag_obj.instructions = self.instructions
        await ag_obj.self_transduction()
        return ag_obj

    # async def self_transduce_batch(self, ag_objects: List[AG]) -> List[AG]:
    #     """Run transduction asynchronously on a batch of agentic objects."""
    #     tasks = [self.transduce(ag) for ag in ag_objects]
    #     results = await asyncio.gather(*tasks)
    #     return results