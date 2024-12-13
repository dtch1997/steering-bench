# Steering Bench

[![arXiv](https://img.shields.io/badge/arXiv-2407.12404-b31b1b.svg)](https://arxiv.org/abs/2407.12404)

Official codebase for the paper: [Analyzing the Generalization and Reliability of Steering Vectors](https://arxiv.org/abs/2407.12404)

## Quickstart
This repository contains instructions on how to run layer sweep and steering experiments described in our paper. 

First, install dependencies: 
```bash
git clone https://github.com/dtch1997/steering-bench
cd steering-bench 
pip install -e . 
```

## Experiments
This repository facilitates running the following experiments: 
- [Layer sweep](experiments/layer_sweep): Extract and apply steering vectors at many different layer in order to select the 'best' layer (by steerability). 
- [Steering generalization](experiments/steering_generalization): Run steering on a given task with variations in user and system prompts to evaluate generalization.

## Package
This repository also provides off-the-shelf components that make it easy to run a custom steering experiment. 
- [Pipeline](steering_bench/core/pipeline), a wrapper around a (possibly-steered) model
- [Formatter](steering_bench/core/format.py), an abstraction for prompt-based scaffolding
- [PipelineHook](steering_bench/core/pipeline), an abstraction for generic steering interventions
- [SteeringHook](steering_bench/core/hook), an implementation of applying steering vectors using our [steering vectors library](https://github.com/steering-vectors/steering-vectors/)
- [Steerability metrics](steering_bench/core/metric.py) used in the paper

## Paper Reproduction
This codebase has been simplified to improve readability, and was not directly used in generating results for the paper. If you would like to reproduce specific plots in our paper, refer to our [original codebase](https://github.com/dtch1997/repepo). 

## Citation
If you found this useful, consider citing our paper: 
```
@misc{tan2024analyzinggeneralizationreliabilitysteering,
      title={Analyzing the Generalization and Reliability of Steering Vectors}, 
      author={Daniel Tan and David Chanin and Aengus Lynch and Dimitrios Kanoulas and Brooks Paige and Adria Garriga-Alonso and Robert Kirk},
      year={2024},
      eprint={2407.12404},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.12404}, 
}
```
## Acknowledgements
This work was made possible by the generous support of:
- FAR AI
- The UCL AI Centre
- UCL DARK Lab
- The Agency for Science, Technology, and Research
