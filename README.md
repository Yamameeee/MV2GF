<div align="center">

# MV2GF: Multi-view Pedestrian Detection <br> with a Visual Geometric Foundation Model

[![Project Page](https://img.shields.io/badge/MV2GF-Website-green?logo=googlechrome&logoColor=green)](https://mv2gf.github.io/)
[![Conference](https://img.shields.io/badge/ECCV-2026-blue)](https://eccv.ecva.net/)
[![arXiv](https://img.shields.io/badge/arXiv-2608.20639-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2608.20639)
</div>


## Updates

- [2026/08]: The sample training code of MV2GF on one sequence in GMVD are available.
- [2026/07]: MV2GF is selected as an oral (spotlight) presentation in ECCV 2026.
- [2026/06]: MV2GF is accepted to ECCV 2026.


## Overview
**MV2GF** is a multi-view pedestrian detection method that leverages a visual geometric foundation model, such as Depth Anything 3 or MapAnything.
By leveraging general-purpose geometric features and predicted 3D pointmaps from the foundation model, MV2GF generalized better to camera configurations not included training data than previous multi-view pedestrian detection methods.
In this repository, we release an sample training code of MV2GF on one sequence in GMVD datasets for understanding our method.


## Quick Start
1. Create a new Conda environment.
    ```shell
    conda create --name mv2gf python=3.12 -y
   ```
2. Install [PyTorch](https://pytorch.org/get-started/locally/) and torchvision with CUDA support.
    ```shell
    pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
   ```
3. Clone this repository and install remaining dependenies.
    ```shell
    cd MV2GF
    pip install -r requirements.txt
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

   ```
4. Download GMVD dataset and `train_datapath.csv` from [this](https://github.com/jeetv/GMVD_dataset) repository. The dataset directory should be like this.
    ```text
    GMVD_DATA
    ├── DATASETS
    │   ├── scene1
    │   │   ├── config1
    |   │   │   ├── seq1
    |   |   │   │   ├── annotations_positions
    |   |   │   │   ├── calibrations
    |   |   │   │   ├── Image_subsets
    |   |   │   │   ├── matchings
    |   |   │   │   ├── config.json
    |   |   │   │   ├── gt.txt
    |   │   │   ├── seq2
    |   │   │   ├── seq3
    |   │   │   ├── seq4
    |   │   │   ├── seq5
    │   ├── scene2
    │   ├── scene3
    │   ├── scene5
    │   ├── scene6
    ├── detect
    │   ├── gta_scene5
    │   │   ├── annotations_positions
    │   │   ├── calibrations
    │   │   ├── Image_subsets
    │   │   ├── matchings
    │   │   ├── config.json
    │   │   ├── gt.txt
    │   ├── unity_scene1
    │   ├── unity_scene2
    ├── train_datapath.csv
    ```
5. Pre-compute and save outputs from [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3) in `npz` format. Please refer to [this](https://github.com/ByteDance-Seed/depth-anything-3) repository for detailed usage of Depth Anything 3. The `npz` file should be saved like this.
    ```python
    import os
    import csv
    save_dir = [PATH to DA3_OUTPUT]
    save_file = os.path.join(
        save_dir,
        'scene1',
        'config1',
        'seq1',
        f'{[FRAME NUMBER]:04}.npz'
    )
    np.savez(
        save_file,
        point=pointmap, # (N, H, W, 3)
        depth=depth, # (N, H, W)
        conf=confidence, # (N, H, W)
        extrinsic=extrinsic, # (N, 3, 4)
        intrinsic=intrinsic, # (N, 3, 3)
        feat_layer_19=feat_layer_19, # (N, H', W', C)
        feat_layer_27=feat_layer_27, # (N, H', W', C)
        feat_layer_33=feat_layer_33, # (N, H', W', C)
        feat_layer_39=feat_layer_39, # (N, H', W', C)
    ) # N is the number of view. H and W are the height and width of images. H' and W' are the height and width of features.
    ```

6. Run training code. Set the path to `train_datapath.csv`, `GMVD_DATA` directory, and `DA3_OUTPUT` directory.
    ```shell
    python trainer.py fit -c config.yml --data.init_args.csv_path [PATH to train_datapath.csv] --data.init_args.data_dir [PATH to GMVD_DATA] --data.init_args.da3_dir [PATH to DA3_OUTPUT]
    ```


## Citation
```bibtex
@inproceedings{yamane2026mv2gf,
    title={MV2GF: Multi-view Pedestrian Detection with a Visual Geometric Foundation Model},
    author={Taiga Yamane and Satoshi Suzuki and Ryo Masumura and Shota Orihashi and Tomohiro Tanaka and Mana Ihori and Naoki Makishima},
    booktitle={ECCV},
    year={2026}
}
```


## License
The code is released under the NTT License as found in the [LICENSE](./LICENSE) file.


## Acknowledgement
- [Simple-BEV](https://simple-bev.github.io): Adam W. Harley
- [MVDeTr](https://github.com/hou-yz/MVDeTr): Yunzhong Hou
- [TrackTacular](https://github.com/tteepe/earlybird): Torben Teepe
- [GMVD](https://github.com/jeetv/GMVD_dataset): Jeet Vora