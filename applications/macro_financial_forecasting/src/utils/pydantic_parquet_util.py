import pandas as pd
from typing import Type
from src.data_model.bloomberg_news_entry import BloombergNewsEntry

class ParquetUtil:
    @staticmethod
    def save_pydantic_to_parquet(objects: list[BloombergNewsEntry], filename: str):
        # Convert list of Pydantic objects to DataFrame by dict conversion
        df = pd.DataFrame([obj.dict() for obj in objects])
        df.to_parquet(filename, index=False)

    @staticmethod
    def save_df_to_parquet(df: pd.DataFrame, filename: str):
        df.to_parquet(filename, index=False)

    @staticmethod
    def load_pydantic_from_parquet(filename: str, model_class: Type[BloombergNewsEntry]) -> list[BloombergNewsEntry]:
        df = pd.read_parquet(filename)
        return [model_class.parse_obj(row) for row in df.to_dict(orient="records")]