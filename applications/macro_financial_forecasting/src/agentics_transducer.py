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

    async def self_transduce(self, ag_obj: AG, instructions: str, source_fields: List[str], target_fields: List[str]) -> AG:
        """Run transduction on a single agentic object."""
        await ag_obj.self_transduction(
            source_fields=source_fields,
            target_fields=target_fields,
            instructions= instructions
        )
        return ag_obj
    
    async def batch_self_transduce(self, ag_objs: List[AG], instructions: str) -> List[AG]:
        # async def transduce_single(ag: AG):
        #     ag.instructions = instructions
        #     await ag.self_transduction()
        #     return ag

        results = await asyncio.gather(
            *(self.self_transduce(ag, instructions) for ag in ag_objs),
            return_exceptions=True
        )
        successful_results = [res for res in results if not isinstance(res, Exception)]
        return successful_results
    
    async def batch_process_with_chunks(self, ag_obj: AG, instructions: str, chunk_size: int = 500) -> List[AG]:
        results = []
        for i in range(0, len(self.states), chunk_size):
            batch = self.states[i:i+chunk_size]
            ag_list = [self.create_AG(ag_obj, [entry]) for entry in batch]
            batch_results = await self.batch_self_transduce(ag_list, instructions)
            results.extend(batch_results)
            # Optionally save intermediate results to disk/db here to free memory
        return results