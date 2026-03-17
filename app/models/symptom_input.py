from pydantic import BaseModel, field_validator
from typing import Optional, List


class SymptomInput(BaseModel):
    symptoms: List[str]
    duration: Optional[str] = None
    severity: Optional[int] = None
    age: Optional[int] = None
    existing_conditions: Optional[List[str]] = None

    @field_validator("severity")
    @classmethod
    def severity_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 10):
            raise ValueError("Severity must be between 1 and 10")
        return v

    @field_validator("age")
    @classmethod
    def age_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (0 <= v <= 120):
            raise ValueError("Invalid age: must be between 0 and 120")
        return v

    @field_validator("symptoms")
    @classmethod
    def symptoms_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one symptom must be provided")
        return v

    def to_prompt_string(self) -> str:
        """Convert structured input to a natural-language string for the pipeline."""
        parts = [f"Symptoms: {', '.join(self.symptoms)}"]
        if self.duration:
            parts.append(f"Duration: {self.duration}")
        if self.severity is not None:
            parts.append(f"Severity: {self.severity}/10")
        if self.age is not None:
            parts.append(f"Age: {self.age}")
        if self.existing_conditions:
            parts.append(f"Existing conditions: {', '.join(self.existing_conditions)}")
        return ". ".join(parts)
