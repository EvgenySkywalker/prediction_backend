from pydantic import BaseModel


class Label(BaseModel):
    start: str
    end: str
    mode: int
