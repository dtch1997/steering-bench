import abc
import torch

from typing import Any
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationConfig

# Base types
@dataclass
class Completion:
    prompt: str
    response: str


@dataclass
class Example:
    positive: Completion
    negative: Completion
    meta: dict[str, Any] | None = None
    steering_token_index: int = -1  # Token index to extract and apply steering vectors

Dataset = list[Example]

Model = PreTrainedModel
Tokenizer = PreTrainedTokenizerBase

class Formatter(abc.ABC):
    """Interface for formatters"""

    @abc.abstractmethod
    def format_as_str(
        self, completion: Completion
    ) -> str:
        pass

    @abc.abstractmethod
    def format_prompt_as_str(
        self, completion: Completion
    ) -> str:
        pass

    @abc.abstractmethod
    def format_conversation(
        self,
        current_message: Completion,
        history: list[Completion] = [],
    ) -> Completion:
        pass

@dataclass
class TokenProb:
    token_id: int
    # Note: the logprobs are for this token, not the next token
    # Recall: logprob(A) - logprob(B) = logit(A) - logit(B)
    logprob: float

@dataclass
class TextProbs:
    """ Utility class to store token-wise logprobs """
    text: str
    token_probs: list[TokenProb]

    @property
    def sum_logprobs(self) -> float:
        return sum([tp.logprob for tp in self.token_probs])

    def __repr__(self) -> str:
        return f"TextProbs({self.text}:{self.sum_logprobs:.2f})"

class Pipeline(abc.ABC):
    """ Abstract interface for a text generation pipeline """

    @abc.abstractmethod
    def build_generation_prompt(self, completion: Completion) -> str:
        """Build the generation prompt from the completion"""
        pass

    @abc.abstractmethod
    def build_full_prompt(self, completion: Completion) -> str:
        """Build the full prompt from the completion"""

    @abc.abstractmethod
    def generate(
        self,
        completion: Completion,
        generation_config: GenerationConfig | None = None,
        remove_base_prompt: bool = True,
    ) -> str:
        pass

    @abc.abstractmethod
    def calculate_output_logprobs(
        self, completion: Completion
    ) -> TextProbs:
        pass