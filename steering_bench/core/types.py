import abc
from typing import Any
from dataclasses import dataclass, field
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationConfig

from inspect_ai.dataset import Sample as _Sample

# Define Completion as an alias for the Sample type
Completion = _Sample


# Define Example as a dataclass with positive and negative completions
@dataclass
class Example:
    positive: Completion
    negative: Completion
    steering_token_index: int = -1  # Token index to extract and apply steering vectors
    # TODO: support steering at all token indices
    meta: dict[str, Any] = field(default_factory=dict)


# TODO: Support 'unpaired' examples

Dataset = list[Example]

Model = PreTrainedModel
Tokenizer = PreTrainedTokenizerBase

Message = dict[str, str]  # keys: 'role', 'content'


class Formatter(abc.ABC):
    """Interface for formatters"""

    @abc.abstractmethod
    def __call__(self, prompt: str) -> list[Message]:
        pass


@dataclass
class TokenProb:
    token_id: int
    # Note: the logprobs are for this token, not the next token
    # Recall: logprob(A) - logprob(B) = logit(A) - logit(B)
    logprob: float


@dataclass
class TextProbs:
    """Utility class to store token-wise logprobs"""

    text: str
    token_probs: list[TokenProb]

    @property
    def sum_logprobs(self) -> float:
        return sum([tp.logprob for tp in self.token_probs])

    def __repr__(self) -> str:
        return f"TextProbs({self.text}:{self.sum_logprobs:.2f})"


class Templater(abc.ABC):
    """Abstract interface for templating a completion as a string to be used as input to a model"""

    @abc.abstractmethod
    def build_generation_prompt(self, completion: Completion) -> str:
        """Build the generation prompt from the completion"""
        pass

    @abc.abstractmethod
    def build_full_prompt(self, completion: Completion) -> str:
        """Build the full prompt from the completion"""


class Pipeline(abc.ABC):
    """Abstract interface for a text generation pipeline"""

    @abc.abstractmethod
    def generate(
        self,
        completion: Completion,
        generation_config: GenerationConfig | None = None,
        remove_base_prompt: bool = True,
    ) -> str:
        pass

    @abc.abstractmethod
    def calculate_output_logprobs(self, completion: Completion) -> TextProbs:
        pass
