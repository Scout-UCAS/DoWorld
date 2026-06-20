# Do-World

Counterfactual world models for intervention-aware planning.

Do-World learns an object-level latent structural world model and uses it for robust planning under counterfactual
interventions. Instead of only predicting the observed future, Do-World supports queries such as object removal,
relation cutting, and local mechanism perturbation, then scores action sequences by comparing factual and
counterfactual rollouts.

## Highlights

- Object-level neural causal mechanism library in slot latent space.
- Editable interventions: `do(remove object)`, `do(cut relation)`, and `do(perturb mechanism)`.
- Counterfactual MPC with factual reward, robust counterfactual reward, and factual-counterfactual gap penalty.
- Training hooks for true intervention consistency, pseudo-intervention supervision, and language-mechanism alignment.
- Evaluation metrics for prediction error, intervention error, counterfactual drop, relation sparsity, and mechanism diversity.
- Continuous-action CEM planning and optional discrete-action categorical CEM planning.

## Project Layout

```text
configs/
  train_do_world.yaml          Do-World training config
  evaluate_do_world.yaml       Do-World evaluation config
sold/
  train_sold.py                Training entry point with Do-World hooks
  evaluate_do_world.py         Do-World metric evaluation
  modeling/sold/do_world.py    Causal mechanism library and Counterfactual MPC
  datasets/do_world.py         Offline NPZ dataset loader for interventions/language fields
  utils/language.py            Dependency-free frozen text encoder for language descriptions
  tests/smoke_do_world.py      Lightweight smoke tests
```

## Installation

Create an environment with PyTorch and the project dependencies. A Conda environment template is available at:

```bash
conda env update -n mof -f apptainer/environment.yml
```

For quick code-level testing, the minimum Python packages are:

```bash
pip install torch torchvision hydra-core lightning gym termcolor tensorboardX numpy pillow tqdm
```

The full online manipulation experiments additionally require the simulator dependencies used by the environment suite.

## Quick Test

Run the smoke test from the repository root:

```bash
PYTHONPATH=sold python sold/tests/smoke_do_world.py
```

Expected output:

```text
do_world smoke tests passed
```

You can also compile the main modules:

```bash
python -m compileall sold/modeling sold/train_sold.py sold/evaluate_do_world.py sold/datasets sold/utils sold/tests
```

## Training

Train the Do-World model with:

```bash
PYTHONPATH=sold python sold/train_sold.py --config-name train_do_world
```

The config uses:

- `modeling.sold.do_world.make_do_world_dynamics_model` for object-level causal dynamics.
- `planning_mode: counterfactual_mpc` for evaluation-time planning.
- `pseudo_intervention_loss_weight` for pseudo object/relation intervention supervision.
- `intervention_loss_weight` for true intervention supervision when intervention fields are present.

## Data Fields

Ordinary online replay data can be used without extra fields. If intervention or language supervision is available,
Do-World will consume the following optional tensor fields from batches or offline NPZ episodes:

```text
intervention_source_slots
intervention_target_slots
intervention_obs
intervention_next_obs
intervention_action
intervention_object_mask
intervention_relation_mask
intervention_mechanism_scale
language_embedding
language_description
mechanism_label
```

Offline NPZ episodes can be loaded with `sold/datasets/do_world.py`. Required keys are `obs` or `images`, and `action`
or `actions`. Optional text descriptions are converted into deterministic embeddings by `sold/utils/language.py`.

## Evaluation

Evaluate Do-World-specific metrics with:

```bash
PYTHONPATH=sold python sold/evaluate_do_world.py checkpoint_path=PATH_TO_CHECKPOINT
```

The evaluator reports:

- episode return and success rate when provided by the environment
- one-step slot prediction error
- multi-step slot prediction error
- true intervention error when intervention targets are available
- factual return
- robust counterfactual return
- counterfactual return drop
- factual-counterfactual gap
- relation sparsity
- mechanism entropy and diversity

## Main Components

### Object-Level Causal Dynamics

`DoWorldDynamicsModel` decomposes slot dynamics into directed relation messages, a learned mechanism router, and a
shared mechanism library. Each object slot is updated by a weighted mixture of local transition mechanisms.

### Interventions

The dynamics model supports intervention dictionaries:

```python
{"object_mask": object_mask}
{"relation_mask": relation_mask}
{"mechanism_scale": mechanism_scale}
```

These can remove objects, cut directed relation edges, or perturb local mechanisms during rollout.

### Counterfactual MPC

`CounterfactualMPCPlanner` samples candidate action sequences, evaluates factual rollouts and counterfactual rollouts,
and optimizes:

```text
score = factual_return + alpha * robust_return - beta * factual_counterfactual_gap
```

For continuous control it uses Gaussian CEM. For discrete domains, pass `discrete_actions` in the planner config to use
categorical CEM over a fixed action table.

## Current Status

The repository contains the complete Do-World implementation layer, training hooks, evaluation script, and smoke tests.
Full benchmark training still depends on installing the target simulator environments and preparing task datasets.
