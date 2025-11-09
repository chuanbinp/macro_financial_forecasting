from config import Config
from train_data_loader import TrainDataLoader

config = Config("../config.env")
train_data_loader = TrainDataLoader(config.dataset_name, config.split_name)

def main():
    print("Starting processing pipeline ...")
    print(f"Industry Types: {config.industries}")
    train_ds = train_data_loader.load()
    print("train_ds sample:", train_ds[101010])
    print("Processing pipeline completed.")

if __name__ == "__main__":
    main()