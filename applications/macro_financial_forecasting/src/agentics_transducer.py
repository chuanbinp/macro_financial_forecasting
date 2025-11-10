from typing import List
from agentics import AG  # as per notebook
from pydantic import BaseModel
from crewai import LLM  # example LLM wrapper
from config import Config
import asyncio


class AgenticTransducer:
    """Encapsulates agentic transduction logic for async LLM calls."""

    def __init__(self, config: Config, verbose: bool = False):
        self.llm = LLM(model=config.llm_model)
        self.verbose = verbose

    def create_AG(self, pydantic_class: BaseModel, data: List[BaseModel]) -> AG:
        return AG(atype=pydantic_class, states=data, llm=self.llm, verbose=self.verbose)

    async def self_transduce(self, ag_obj: AG, instructions: str) -> AG:
        """Run transduction on a single agentic object."""
        ag_obj.instructions = instructions
        await ag_obj.self_transduction()
        return ag_obj
    
    async def batch_self_transduce(self, ag_objs: List[AG], instructions: str) -> List[AG]:
        async def transduce_single(ag: AG):
            ag.instructions = instructions
            await ag.self_transduction()
            return ag

        results = await asyncio.gather(
            *(transduce_single(ag) for ag in ag_objs),
            return_exceptions=True
        )

        successful_results = [res for res in results if not isinstance(res, Exception)]

        return successful_results