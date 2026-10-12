"""Request and response bodies, validated by Pydantic."""

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_CHARS = 500  # the longest CLINC150 query is under 150 characters


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS, examples=["what's my balance"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must contain something other than whitespace")
        return value


class IntentScore(BaseModel):
    intent: str
    confidence: float


class ClassifyResponse(BaseModel):
    label: str  # an intent name, or "out_of_scope"
    confidence: float  # probability of the top intent
    top_intents: list[IntentScore]  # the three most likely intents, highest first


class ModelInfo(BaseModel):
    model: str
    base_model: str
    num_labels: int
    oos_threshold: float
    max_text_chars: int
