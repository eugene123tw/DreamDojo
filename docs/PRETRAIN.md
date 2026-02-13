# DreamDojo Pretraining

Suppose you are training on 1 node with 8 GPUs:

1. Make sure you have set the correct paths in `launch.sh`.
2. Replace the `ckpt_path` in `cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py` to the actual latent action model path.
3. Launch DreamDojo pretraining by:
    
    ```bash
    bash launch.sh dreamdojo_2b_480_640_pretrain
    ```

---

<= Previous: [[Latent Action Model Training](https://github.com/NVIDIA/DreamDojo/blob/main/docs/LAM.md)]

=> Next: [[DreamDojo Post-Training](https://github.com/NVIDIA/DreamDojo/blob/main/docs/POSTTRAIN.md)]