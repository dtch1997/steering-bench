from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from transformers.generation import GenerationConfig
from jaxtyping import Float, Int
from eindex import eindex

import torch

from .types import Completion, Model, Tokenizer
from .format import Formatter


@dataclass
class TokenProb:
    token_id: int
    # Note: the logit, logprob are for this token, not the next token
    logprob: float
    text: str | None = None


@dataclass
class TextProbs:
    text: str
    token_probs: list[TokenProb]

    @property
    def sum_logprobs(self) -> float:
        return sum([tp.logprob for tp in self.token_probs])

    def __repr__(self) -> str:
        return f"TextProbs({self.text}:{self.sum_logprobs:.2f})"


@dataclass
class PipelineContext:
    """Dataclass to store context information for hooks.

    The pipeline context tells hooks about how the pipeline is being called.
    Crucially, this is the only way that hooks know about individual prompts
    """

    method: Literal["generate", "logprobs"]
    base_prompt: str
    full_prompt: str
    inputs: Any
    pipeline: "Pipeline"


class PipelineHook(Protocol):
    def __call__(self, context: PipelineContext) -> AbstractContextManager[None]: ...


@dataclass
class Pipeline:
    """A lightweight pipeline for generating completions and calculating logprobs.

    Wraps a Huggingface model and tokenizer.
    Supports hooks for running modified inference.
    """

    model: Model
    tokenizer: Tokenizer
    formatter: Formatter
    hooks: list[PipelineHook] = field(default_factory=list)

    # NOTE: currently unused
    _print_first_example: bool = True

    @contextmanager
    def append_hooks(self, hooks: list[PipelineHook]) -> Any:
        """A context manager for adding temporary hooks to the model."""
        orig_hooks = self.hooks
        try:
            self.hooks = orig_hooks + hooks
            yield
        finally:
            # NOTE: Also resets hooks that were added manually inside the context
            self.hooks = orig_hooks

    def generate(
        self,
        completion: Completion,
        generation_config: GenerationConfig | None = None,
        remove_base_prompt: bool = True,
    ) -> str:
        """Generate a completion for a given example"""
        # Setup
        formatted_completion = self.formatter.format(completion)
        del completion
        base_prompt = formatted_completion.prompt
        full_prompt = formatted_completion.as_str()

        inputs: Any = self.tokenizer(base_prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        context = PipelineContext(
            method="generate",
            base_prompt=base_prompt,
            full_prompt=full_prompt,
            inputs=inputs,
            pipeline=self,
        )

        # Main logic
        with ExitStack() as stack:
            # Apply hooks
            for hook in self.hooks:
                stack.enter_context(hook(context))
            # Generate the outputs
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )[0]

        # Create the output
        outputs_str = self.tokenizer.decode(outputs, skip_special_tokens=True)

        # Optionally, remove the base prompt from the outputs (for QoL)
        if remove_base_prompt:
            return outputs_str.replace(base_prompt, "")
        return outputs_str

    @torch.no_grad()
    def logprobs(self, completion: Completion) -> TextProbs:
        """Calculate the logprobs for each token in the prompt + output"""

        # Setup
        formatted_completion = self.formatter.format(completion)
        del completion
        base_prompt = formatted_completion.prompt
        full_prompt = formatted_completion.as_str()

        inputs: Any = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        context = PipelineContext(
            method="logprobs",
            base_prompt=base_prompt,
            full_prompt=full_prompt,
            inputs=inputs,
            pipeline=self,
        )

        # Main logic
        with ExitStack() as stack:
            for hook in self.hooks:
                stack.enter_context(hook(context))
            outputs = self.model(**inputs, output_hidden_states=False, return_dict=True)
            logprobs = torch.log_softmax(outputs.logits, dim=-1)

        # get the logprob of the generated token
        # NOTE: Be very careful about off-by-one indexing error!
        # -- logprob at index 0 corresponds to the token at index 1
        next_logprobs: Float[torch.Tensor, "batch seq d_vocab"] = logprobs[:, :-1, :]
        next_tokens: Int[torch.Tensor, "batch seq"] = inputs.input_ids[:, 1:]
        logprobs_of_generated_tokens: Float[torch.Tensor, "batch seq"] = eindex(
            next_logprobs, next_tokens, "batch seq [batch seq]"
        )
        assert (
            logprobs_of_generated_tokens.shape == next_tokens.shape
        ), f"Expected logprobs of shape {next_tokens.shape}; got {logprobs_of_generated_tokens.shape}"
        assert (
            logprobs_of_generated_tokens.shape[0] == 1
        ), f"Expected batch size of 1, got {logprobs_of_generated_tokens.shape[0]}"

        # Create the output
        text_probs: list[TokenProb] = []
        for token, logprob in zip(
            next_tokens.squeeze(0),
            logprobs_of_generated_tokens.squeeze(0),
        ):
            if token in self.tokenizer.all_special_ids:
                continue

            token_id = token.item()
            assert isinstance(token_id, int)
            token_prob = TokenProb(
                token_id=token_id,
                logprob=logprob.item(),
            )
            text_probs.append(token_prob)
        return TextProbs(text=full_prompt, token_probs=text_probs)

    def build_full_prompt(self, completion: Completion) -> str:
        """Syntactic sugar to build a full prompt from a completion"""
        return self.formatter.format(completion).as_str()
