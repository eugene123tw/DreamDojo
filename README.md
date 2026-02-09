<img src="assets/banner.gif" width="100%"/>

<div align="center">
  <p style="font-size: 1.2em;">
    <a href="https://dreamdojo-world.github.io/"><strong>Website</strong></a> | 
    <a href="https://arxiv.org/abs/2602.06949"><strong>Paper</strong></a> |
    <strong>Models</strong> |
    <strong>Eval Sets</strong>
  </p>
</div>

# 💭 DreamDojo

## 🔥 Highlights

*DreamDojo* is an interactive world model that learns from large-scale human videos. In short, we made the following key contributions:

- **A large-scale video dataset.** 44k hours of diverse human egocentric videos, the largest dataset to date for world model pretraining.
- **A foundation world model.** The first robot world model of its kind that demonstrates strong generalization to diverse objects and environments after post-training.
- **A distillation pipeline.** After distillation, our model can achieve long-horizon autoregressive generation, with stable real-time interactions at 10 FPS for over 1 minute.

## 📢 News

- **[2026/02/09]** We released both pretraining and post-training code.
- **[2026/02/09]** We released our [paper](https://arxiv.org/abs/2602.06949) on arXiv.

## 📋 Release Plan

We aim to release the following items this month.

- [ ] DreamDojo-14B model config.
- [ ] Latent action model and DreamDojo checkpoints.
- [ ] Evaluation sets.
- [ ] Distillation pipeline.
- [ ] Teleoperation code.

## 🕹️ Quick Start

### Installation

The current code is tested with [uv](https://docs.astral.sh/uv/). To install the environment with uv:

```bash
bash install.sh
```

### Latent Action Model Training

Suppose you are training on 1 node with 8 GPUs:

```bash
cd external/lam_project
bash train.sh
```

A multi-node training script example is provided at `external/lam_project/launch.sh`.

### DreamDojo Pretraining

Suppose you are training on 1 node with 8 GPUs:

1. Make sure you have set the correct paths in `launch.sh`.
2. Replace the `ckpt_path` in `cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py` to the actual latent action model path.
3. Launch DreamDojo pretraining by:
    
    ```bash
    bash launch.sh groot_ac_reason_embeddings_rectified_flow_2b_480_640_pretrain
    ```

### DreamDojo Post-Training

The training configurations are managed by yaml under `configs`. To launch post-training:

1. Follow the same setup as the pretraining.
2. Change `load_path` in `cosmos_predict2/experiments/base/action.py` to the pretrained checkpoint and specify its training step.
3. (Optional) Disable the extraction of latent actions on the fly, as it will be reset to zero.
4. Send the desired experiment config (e.g., `groot_ac_reason_embeddings_rectified_flow_2b_480_640_gr1` to use `ab_480_640_gr1.yaml`).
    
    ```bash
    bash launch.sh groot_ac_reason_embeddings_rectified_flow_2b_480_640_gr1
    bash launch.sh groot_ac_reason_embeddings_rectified_flow_2b_480_640_g1
    bash launch.sh groot_ac_reason_embeddings_rectified_flow_2b_480_640_agibot
    bash launch.sh groot_ac_reason_embeddings_rectified_flow_2b_480_640_yam
    ```

### Evaluation

When adapting to the target robot action space through post-training, you can keep tracking the performance of the checkpoints at a specified interval:

```bash
bash eval.sh
```

</details>

## ⭐ Citation

If you find our work useful, please consider citing us and giving a star to our repo.

```bibtex
@article{gao2026dreamdojo,
    title={DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos},
    author={Shenyuan Gao and William Liang and Kaiyuan Zheng and Ayaan Malik and Seonghyeon Ye and Sihyun Yu and Wei-Cheng Tseng and Yuzhu Dong and Kaichun Mo and Chen-Hsuan Lin and Qianli Ma and Seungjun Nah and Loic Magne and Jiannan Xiang and Yuqi Xie and Ruijie Zheng and Dantong Niu and You Liang Tan and K.R. Zentner and George Kurian and Suneel Indupuru and Pooya Jannaty and Jinwei Gu and Jun Zhang and Jitendra Malik and Pieter Abbeel and Ming-Yu Liu and Yuke Zhu and Joel Jang and Linxi "Jim" Fan},
    journal={arXiv preprint arXiv:2602.06949},
    year={2026}
}
```

## ⚖️ License

DreamDojo source code is released under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).