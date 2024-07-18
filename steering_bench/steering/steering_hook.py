from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal
from steering_vectors import (
    SteeringVector,
    ModelLayerConfig,
    ablation_then_addition_operator,
    addition_operator,
    PatchDeltaOperator,
)

from steering_bench.core.pipeline import PipelineContext
from steering_bench.core.types import Tokenizer
from steering_bench.core.pipeline import PipelineHook


@dataclass(frozen=True)
class SteeringConfig:
    layer: int
    multiplier: float = 1.0
    patch_generation_tokens_only: bool = True
    skip_first_n_generation_tokens: int = 0
    layer_config: ModelLayerConfig | None = None
    patch_operator: Literal["add", "ablate_then_add"] = "add"


@dataclass(frozen=True)
class SteeringHook(PipelineHook):
    """
    Pipeline hook that applies a steering vector to the model.
    All relevant state for the hook is stored in this class.
    If the params included in this class are changed, it will
    affect future generation and logprob calls using this hook.
    """

    steering_vector: SteeringVector
    config: SteeringConfig

    # PipelineContext is created in both `pipeline.generate` or `pipeline.calculate_output_logprobs`,
    # It also contains info about the current prompt which is used to determine which tokens to patch.
    @contextmanager
    def __call__(self, context: PipelineContext):
        handle = None
        try:
            layer_sv = get_single_layer_steering_vector(
                self.steering_vector, self.config.layer
            )

            # get the index where we should start patching
            min_token_index = 0
            if self.config.patch_generation_tokens_only:
                gen_start_index = get_generation_start_token_index(
                    context.pipeline.tokenizer,
                    context.base_prompt,
                    context.full_prompt,
                )
                min_token_index = (
                    gen_start_index + self.config.skip_first_n_generation_tokens
                )

            # multiplier 0 is equivalent to no steering, so just skip patching in that case
            if self.config.multiplier == 0:
                yield

            handle = layer_sv.patch_activations(
                model=context.pipeline.model,
                layer_config=self.config.layer_config,
                multiplier=self.config.multiplier,
                min_token_index=min_token_index,
                operator=get_patch_operator(self.config.patch_operator),
            )
            yield

        finally:
            if handle is not None:
                handle.remove()


def get_single_layer_steering_vector(
    steering_vector: SteeringVector, layer: int
) -> SteeringVector:
    """Returns a new steering vector that only modifies the specified layer"""
    assert (
        layer in steering_vector.layer_activations
    ), f"Layer {layer} not found in steering vector"
    new_layer_activations = {layer: steering_vector.layer_activations[layer]}
    return SteeringVector(
        layer_activations=new_layer_activations,
        layer_type=steering_vector.layer_type,
    )


def get_patch_operator(
    patch_operator: Literal["add", "ablate_then_add"],
) -> PatchDeltaOperator:
    """Returns the operator function for the specified patch operator"""
    if patch_operator == "ablate_then_add":
        return ablation_then_addition_operator()
    elif patch_operator == "add":
        return addition_operator()
    else:
        raise ValueError(f"Unknown patch operator: {patch_operator}")


def get_generation_start_token_index(
    tokenizer: Tokenizer, base_prompt: str, full_prompt: str
) -> int:
    """Finds the index of the first generation token in the prompt"""
    base_toks = tokenizer.encode(base_prompt)
    full_toks = tokenizer.encode(full_prompt)
    prompt_len = len(base_toks)
    # try to account for cases where the final prompt token is different
    # from the first generation token, ususally weirdness with spaces / special chars
    for i, (base_tok, full_tok) in enumerate(zip(base_toks, full_toks)):
        if base_tok != full_tok:
            prompt_len = i
            break
    # The output of the last prompt token is the first generation token
    # so need to subtract 1 here
    return prompt_len - 1
