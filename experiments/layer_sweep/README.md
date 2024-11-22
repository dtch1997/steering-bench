Workflow to run layer sweeps and produce layer response curves (of steerability). 

```bash

python download_and_preprocess_datasets.py
# Evaluate propensity across layers
python run_layer_sweep.py
# Plot steerability and propensity curves
python analysis.py
```