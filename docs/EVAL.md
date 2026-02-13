# Evaluation

When/After adapting to the target robot action space through post-training, you can keep tracking the performance of the checkpoints at a specified interval.

```
bash eval.sh
```

Important arguments:

- `--checkpoints-dir`: Path to the saved checkpoints.
- `--experiment`: Configuration name.
- `--save-dir`: Path to save the generated videos.
- `--num-frames`: Length of the ground truth video, including the first condition frame.
- `--num-samples`: Number of the samples to evaluate on.
- `--dataset-path`: Path to the evaluation datasets. You can also concatenate multiple evaluation sets here by spliting them with commas (e.g. `--dataset-path "datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval,datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval,datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval"`).
- `--data-split`: Which subset for evaluation. (`full`: sample from 100%; `train`: the first 95% of each concatenated eval set; `test`: the last 5% of each concatenated eval set).
- `--deterministic-uniform-sampling`: If enabled, sample from the concatenated datasets uniformly instead of by their lengths.
- `--checkpoint-interval`: Interval of the checkpoints to evaluate.
- `--infinite`: Continuously check for new checkpoints at the desired interval to track performance during training.

---

<= Previous: [[DreamDojo Post-Training](https://github.com/NVIDIA/DreamDojo/blob/main/docs/POSTTRAIN.md)]

=> Next: [[Trouble Shooting](https://github.com/NVIDIA/DreamDojo/blob/main/docs/ISSUES.md)]