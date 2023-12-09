Code for Paper [Grounded Reinforcement Learning: Learning to Win the Game under Human Commands](https://drive.google.com/file/d/1ijDwOF3l4qkI7baI84IxA_HL-jf5jn4Z/view?usp=sharing)

Model: `models/single_action_inherit.py`

Evaluation Code: `eval_6_enemy.py`

# Installation

We provide a docker image `jwhj/red:v1.0.0`.

After pulling the image, start a container by
```
docker run -it --gpus 1 --ipc=host --network=host jwhj/red:v1.0.0
```

# Update to Latest

In the directory `/workspace/RED`, run
```
git remote set-url origin https://github.com/jwhj/RED.git
git pull
```
to get the latest code.

# Downloading Dataset and Model Checkpoints

First download the `minirts` dataset and pretrained models.
```
cd minirts/data
sh download.sh
cd ../pretrained_models
sh download.sh
python update_path.py
```

Download checkpoints of the RED policy and the baselines from <https://drive.google.com/file/d/1dEjG9IOCwunUjVUqOOdDxaYr0lmg9Li1/view?usp=sharing>. Put all `.pt` files under the `ckpt/` directory.

# Evaluation
## Evaluate Model Win Rates

Using the Oracle commander:
```
python eval_6_enemy.py --exper-name [EXPERIMENT_NAME] --model-path [MODEL_PATH] --coach-name rule-based
```

Using the Adversarial Oracle commander:
```
python eval_6_enemy.py --exper-name [EXPERIMENT_NAME] --model-path [MODEL_PATH] --coach-name rule-based --adv-coach
```

Using the NA commander:
```
python eval_6_enemy.py --exper-name [EXPERIMENT_NAME] --model-path [MODEL_PATH] --coach-name rule-based --dropout 1
```

Using the Random commander:
```
python eval_6_enemy.py --exper-name [EXPERIMENT_NAME] --model-path [MODEL_PATH] --coach-name random --p [PERCENTAGE_OF_RANDOM_COMMANDS]
```

Using the Human Proxy commander:
```
python eval_6_enemy.py --exper-name [EXPERIMENT_NAME] --model-path [MODEL_PATH]
```

If you are runnning the `switch` baseline, `[EXPERIMENT_NAME]` should be a string containing "switch". Otherwise `[EXPERIMENT_NAME]` may be chosen arbitrarily.

## Evaluate Action Prediction Accuracy Difference between RED and BC

```
python acc.py --compute --plot
```

Similarly, to evaluate the validation NLL between RED and BC, run
```
python nll.py --compute --plot
```

# Gameplay

Run
```
uvicorn human_coach_server:app --port 8000
```

Then access <http://localhost:8000/new_game/[MODEL_NAME]>. Here `[MODEL_NAME]` can be chosen from `red`, `joint`, `switch`, `rl` and `irl`.