# from typing import List
# from agentics import AG  # as per notebook
# from pydantic import BaseModel
# from crewai import LLM  # example LLM wrapper
# from config import Config
# import asyncio
# from pathlib import Path


# class AgenticTransducer:
#     """Encapsulates agentic transduction logic for async LLM calls."""

#     def __init__(self, config: Config, verbose: bool = False):
#         self.llm = LLM(model=config.llm_model)
#         self.verbose = verbose
#         self.data_dir = Path(config.dataset_dir)

#     def create_AG(self, pydantic_class: BaseModel, data: List[BaseModel]) -> AG:
#         return AG(atype=pydantic_class, states=data, llm=self.llm, verbose=self.verbose)

#     async def self_transduce(self, ag_obj: AG, instructions: str, source_fields: List[str], target_fields: List[str]) -> AG:
#         """Run transduction on a single agentic object."""
#         await ag_obj.self_transduction(
#             source_fields=source_fields,
#             target_fields=target_fields,
#             instructions=instructions
#         )
#         return ag_obj
    
#     async def batch_self_transduce(self, ag_objs: List[AG], instructions: str, source_fields: List[str], target_fields: List[str]) -> List[AG]:
#         # async def transduce_single(ag: AG):
#         #     ag.instructions = instructions
#         #     await ag.self_transduction()
#         #     return ag

#         results = await asyncio.gather(
#             *(self.self_transduce(ag, instructions, source_fields, target_fields) for ag in ag_objs),
#             return_exceptions=True
#         )
#         successful_results = [res for res in results if not isinstance(res, Exception)]
#         return successful_results
    
#     async def batch_process_with_chunks(self, pydantic_class: BaseModel, instructions: str, source_fields: List[str], target_fields: List[str], chunk_size: int = 2000) -> List[AG]:
#         results = []
#         for i in range(0, len(self.states), chunk_size):
#             batch = self.states[i:i+chunk_size]
#             ag_list = [self.create_AG(pydantic_class, [entry]) for entry in batch]
#             batch_results = await self.batch_self_transduce(ag_list, instructions, source_fields, target_fields)
#             results.extend(batch_results)
#             # Optionally save intermediate results to disk/db here to free memory
#             self.save_AG(
#                 self.create_AG(pydantic_class, batch_results),
#                 f"transduction_results_{i}.csv"
#             )
#         return results
    
#     def save_AG(self, ag_obj: AG, filename: str) -> None:
#         ag_obj.to_csv(self.data_dir / filename)

#     def load_AG(self, filename: str) -> AG:
#         return AG.from_csv(self.data_dir / filename)
