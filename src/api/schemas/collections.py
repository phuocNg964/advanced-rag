from pydantic import BaseModel, Field

class CollectionCreate(BaseModel):
    name: str = Field(
        ..., 
        min_length=1, 
        max_length=100,
        pattern=r"^[A-Z][a-zA-Z0-9_]*$",
        description="Weaviate collection names must start with a capital letter and contain only alphanumeric characters and underscores."
    )
