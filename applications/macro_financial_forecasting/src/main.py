from config import Config
from train_data_loader import TrainDataLoader
from agentics_transducer import AgenticTransducer
from data_model.bloomberg_news_entry import BloombergNewsEntry
import asyncio

config = Config("../config.env")

train_data_loader = TrainDataLoader(config)
agentic_transducer = AgenticTransducer(config)

async def main():
  print("Starting processing pipeline ...")
  # print(f"Industry Types: {config.industries}")
  
  train_ds = train_data_loader.load()
  print("Processing pipeline completed.")

  ag = agentic_transducer.create_AG(BloombergNewsEntry, train_ds[:10])  # Example usage
  ag = await agentic_transducer.self_transduce(ag, config.prompt_instructions)
  ag.pretty_print()

    
if __name__ == "__main__":
    asyncio.run(main())