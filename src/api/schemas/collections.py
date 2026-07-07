from pydantic import BaseModel, Field

from src.core.names import COLLECTION_NAME_PATTERN


class CollectionCreate(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=COLLECTION_NAME_PATTERN.pattern,
        description="Weaviate collection names must start with a capital letter and contain only alphanumeric characters and underscores.",
    )
