# DreamDojo Post-Training

The training configurations are managed by yaml under `configs`. To launch post-training:

1. Follow the same setup as the pretraining.
2. Change `load_path` in `cosmos_predict2/experiments/base/action.py` to the pretrained checkpoint and specify its training step.
3. (Optional) Disable the extraction of latent actions on the fly, as it will be reset to zero.
4. Send the desired experiment config (e.g., `dreamdojo_2b_480_640_gr1` to use `ab_480_640_gr1.yaml`).
    
    ```bash
    bash launch.sh dreamdojo_2b_480_640_gr1
    bash launch.sh dreamdojo_2b_480_640_g1
    bash launch.sh dreamdojo_2b_480_640_agibot
    bash launch.sh dreamdojo_2b_480_640_yam
    ```

---

<= Previous: [[DreamDojo Pretraining](https://github.com/NVIDIA/DreamDojo/blob/main/docs/PRETRAIN.md)]

=> Next: [[Evaluation](https://github.com/NVIDIA/DreamDojo/blob/main/docs/EVAL.md)]