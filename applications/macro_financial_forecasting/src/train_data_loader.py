from datasets import load_dataset, DatasetDict, Value
from src.config import Config
import os
from src.data_model.bloomberg_news_entry import BloombergNewsEntry
from src.utils.pydantic_parquet_util import ParquetUtil
from tqdm import tqdm

class TrainDataLoader:
    def __init__(self, config: Config) -> None:
        self.dataset_name = config.dataset_name
        self.split_name = "train"
        self.dataset = None
        # Create data folder path and dataset cache path
        self.data_dir = config.dataset_dir
        os.makedirs(self.data_dir, exist_ok=True)
        safe_dataset_name = self.dataset_name.replace("/", "_")
        self.cache_path = os.path.join(self.data_dir, f"{safe_dataset_name}_{self.split_name}")

    def _download_dataset(self) -> DatasetDict:
        print(f"Downloading dataset '{self.dataset_name}' with split '{self.split_name}' from the Hugging Face Hub...")
        try:
            self.dataset = load_dataset(self.dataset_name, split=self.split_name, download_mode="force_redownload")
            print("\n--- Download Successful! ---")
            return self.dataset
        except FileNotFoundError:
            print(f"Error: Dataset or split '{self.dataset_name}/{self.split_name}' not found on the Hub.")
            print("Please check the dataset name and split name for typos.")
        except Exception as e:
            print(f"An unexpected error occurred during dataset loading: {e}")

    def _convert_datetime_to_date_str(self) -> None:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call 'load()' first.")
        new_features = self.dataset.features.copy()
        new_features["Date"] = Value("string")
        self.dataset = self.dataset.map(
            lambda x: {"Date": x["Date"].strftime("%Y-%m-%d")},
            features=new_features,
        )

    def _validate_dataset_entries(self) -> None:
        print(f"\n--- Starting validation of {len(self.dataset)} entries ---")
        validated_ds = [BloombergNewsEntry.model_validate(record) for record in tqdm(self.dataset, desc="Validating entries")]
        print("--- Validation Complete! ---")
        self.dataset = validated_ds

    def load(self) -> DatasetDict:
        # Try loading from saved local dataset first
        if os.path.exists(self.cache_path):
            print(f"Loading dataset from local cache at '{self.cache_path}'...")
            self.dataset = ParquetUtil.load_pydantic_from_parquet(self.cache_path, BloombergNewsEntry)
        else:
            self._download_dataset()
            self._convert_datetime_to_date_str()
            print("Training dataset processed.")
            self._validate_dataset_entries()
            print("Training dataset validated.")
            ParquetUtil.save_pydantic_to_parquet(self.dataset, self.cache_path)
            print(f"Saving processed dataset to local cache at '{self.cache_path}'...")

        print(f"Total number of rows: {len(self.dataset)}")
        return self.dataset