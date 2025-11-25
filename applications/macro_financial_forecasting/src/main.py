from processor import NewsProcessor
from config import Config
from train_data_loader import TrainDataLoader
# from agentics_transducer import AgenticTransducer
# from data_model.bloomberg_news_entry import BloombergNewsEntry
import asyncio

config = Config("../config.env")

train_data_loader = TrainDataLoader(config)
transducer = NewsProcessor(config)

async def main():
  print("Starting processing pipeline ...")
  # print(f"Industry Types: {config.industries}")

  train_ds = train_data_loader.load()
  train_ds = await transducer.process_news_entries_async(train_ds, config.prompt_instructions, save_path_prefix="processed_news")
  print("Processing pipeline completed.")

  print(train_ds[:2])  # Print first 2 entries for verification
  # ag = agentic_transducer.create_AG(BloombergNewsEntry, train_ds[:10])  # Example usage
  # ag = await agentic_transducer.self_transduce(ag, config.prompt_instructions)
  # ag.pretty_print()

    
if __name__ == "__main__":
    asyncio.run(main())