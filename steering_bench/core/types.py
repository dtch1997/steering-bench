from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizerBase


Model = PreTrainedModel
Tokenizer = PreTrainedTokenizerBase


@dataclass(frozen=True)
class Completion:
    prompt: str
    response: str

    def as_str(self) -> str:
        return f"{self.prompt} {self.response}"


@dataclass(frozen=True)
class ContrastivePair:
    prompt: str
    positive_response: str
    negative_response: str

    @property
    def positive_completion(self) -> Completion:
        return Completion(self.prompt, self.positive_response)

    @property
    def negative_completion(self) -> Completion:
        return Completion(self.prompt, self.negative_response)


Dataset = list[ContrastivePair]
