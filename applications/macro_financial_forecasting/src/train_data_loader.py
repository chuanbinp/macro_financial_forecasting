from datasets import load_dataset, DatasetDict, Value


class TrainDataLoader:
    def __init__(self, dataset_name: str, split_name: str):
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.dataset = None

    def _download_dataset(self) -> DatasetDict:
        try:
            dataset = load_dataset(self.dataset_name, split=self.split_name)
            print("\n--- Download Successful! ---")
            print(f"Loaded dataset type: {type(dataset)}")
            print(f"\nTotal number of rows in the '{self.split_name}' split: {len(dataset)}")
            print("\nFeatures (columns) in the dataset:")
            print(dataset.column_names)
            self.dataset = dataset
            return dataset
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

    def load(self) -> DatasetDict:
        self._download_dataset()
        self._convert_datetime_to_date_str()
        print("Training dataset loaded and processed.")
        return self.dataset