from typing import Literal
from pydantic import BaseModel, Field
from config import Config

config = Config("../config.env")
IndustryType = Literal[tuple(config.industries)]

class IndustryAndKeyPoints(BaseModel):
    Industry: IndustryType = Field(description=f"The primary industry sector this news is relevant to. Must be one of: {config.industries}.")
    KeyPoints: str = Field(description="A bullet list summarizing the 5 most important points of the news article.")