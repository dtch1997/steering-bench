# Load model, tokenizer, formatter and build pipeline

import torch
from steering_bench.steering import SteeringConfig
from steering_bench.steering.evaluate_steering_vector import (
    evaluate_steering_vector_sweep,
    make_sweep_layers_and_multipliers,
)

from steering_vectors import train_steering_vector
from steering_bench.data import load_dataset, DatasetSpec
from steering_bench.core.pipeline import Pipeline
from steering_bench.core.format import ChatFormatter
from steering_bench.steering.train_steering_vector import (
    build_steering_vector_training_data,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch.set_grad_enabled(False)
device = "cuda"


def get_model_and_tokenizer(model_name: str):
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, device=device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    return model, tokenizer


def test_train_and_evaluate_steering_vector():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    train_spec = DatasetSpec("xrisk/coordinate-itself", "0%:+3")
    eval_spec = DatasetSpec("xrisk/coordinate-itself", "50%:+3")
    layer = 13
    steering_token_index = -2

    train_dataset = load_dataset(train_spec)
    eval_dataset = load_dataset(eval_spec)

    model, tokenizer = get_model_and_tokenizer(model_name)
    formatter = ChatFormatter(tokenizer)
    pipeline = Pipeline(model, tokenizer, formatter)

    steering_config = SteeringConfig(
        layer=layer,
        multiplier=0,
        skip_first_n_generation_tokens=1,
    )

    steering_vector_training_data = build_steering_vector_training_data(
        pipeline,
        train_dataset,
        steering_token_index=steering_token_index,
    )

    steering_vector = train_steering_vector(
        pipeline.model,
        pipeline.tokenizer,
        steering_vector_training_data,
        layers=[layer],
        show_progress=False,
    )

    sweep = make_sweep_layers_and_multipliers(
        config=steering_config,
        layers=[layer],
        multipliers=[-1.0, 0, 1.0],
    )

    results = evaluate_steering_vector_sweep(
        sweep,
        pipeline,
        steering_vector,
        eval_dataset,
    )
