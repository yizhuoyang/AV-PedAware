# AV-PedAware

This is the repository for "AV-PedAware: Self-Supervised Audio-Visual Fusion for Dynamic Pedestrian Awareness"

<img src="https://github.com/yizhuoyang/AV-PedAware/blob/main/figs/detection_result.gif" width="70%">

## Data
Some of the newly collected data with 8 mic array can be download from this [link](https://pan.baidu.com/s/1VzQnecSW_UPeBkFju6Zf9A?pwd=2024) or [google drive](https://drive.google.com/drive/folders/1nPTepGdy6jtVHSYqUx5XLJW_u6YSFxYd?usp=drive_link)

### Generate paired samples from ROS 2 bags

The script `data_processing/build_pairs_from_rosbags.py` creates timestamp-aligned multimodal samples from extracted wav files and the corresponding ROS 2 bags.

Before running it, make sure:
- extracted wav files are stored under `data/wav/`
- `data/wav/audio_timestamps.csv` contains the mapping from each wav file to its ROS 2 bag
- the source bags contain these topics:
  - `/camera/color/image_raw/compressed`
  - `/camera/depth/image_raw/compressedDepth`
  - `/livox/lidar`

Run:

```bash
$ source /opt/ros/galactic/setup.bash
$ /usr/bin/python3.8 data_processing/build_pairs_from_rosbags.py --overwrite
```

By default, audio is split into `0.5 s` segments. For each segment, the nearest RGB image, depth image, and lidar frame are selected by timestamp and written to:

```text
data/pairs/
  bag_name/
    audio/
      0001.wav
    image/
      0001.png
    depth/
      0001.png
    lidar/
      0001.bin
    manifest.csv
```

Useful options:

```bash
$ /usr/bin/python3.8 data_processing/build_pairs_from_rosbags.py --bag-name rosbag2_2026_05_17-09_34_07 --overwrite
$ /usr/bin/python3.8 data_processing/build_pairs_from_rosbags.py --segment-seconds 0.5 --max-sync-gap-seconds 0.25 --overwrite
```

### Generate paired samples from ROS 1 bags

The ROS 1 version reads `.bag` files together with wav files whose names contain
the audio start timestamp:

```text
wav_dir/
  my_recording_1747800000.125.wav
bag_dir/
  my_recording.bag
```

The final token in the wav stem is interpreted as ROS time in seconds by
default auto-detection. Integer timestamps in nanoseconds are also supported.
The script splits audio into `0.5 s` windows and writes the same
`audio/image/depth/lidar/manifest.csv` pairs layout as the ROS 2 pipeline.
By default, the hop equals the window length. Set `--hop-seconds` for
overlapping windows, for example a `0.5 s` window every `0.1 s`.

```bash
$ source /opt/ros/noetic/setup.bash
$ python3 data_processing/build_pairs_from_ros1_bags.py \
    --wav-dir /path/to/wav_dir \
    --bag-dir /path/to/ros1_bags \
    --output-root data/pairs_ros1 \
    --overwrite
```

The default ROS 1 topics match the existing extraction pipeline:

```text
/camera/color/image_raw/compressed
/camera/depth/image_raw/compressedDepth
/livox/lidar
```

Topics and synchronization behavior can be changed when needed:

```bash
$ python3 data_processing/build_pairs_from_ros1_bags.py \
    --wav-dir /media/kemove/T9/bag/static/wav_exports \
    --bag-dir /media/kemove/T9/bag/static/bag \
    --output-root data/pairs_ros1 \
    --timestamp-unit auto \
    --header-stamp \
    --segment-seconds 0.5 \
    --hop-seconds 0.25 \
    --max-sync-gap-seconds 0.25 \
    --overwrite
```

LiDAR messages are decoded from either Livox `CustomMsg` (`points` containing
`x/y/z/reflectivity`) or standard `sensor_msgs/PointCloud2`, and saved as
float32 `x, y, z, intensity` `.bin` files.

### Generate pseudo 3D bbox labels

Pseudo 3D bounding boxes are generated with a LiDAR-only detector trained from a small set of manually checked sequences.

#### 1. Prepare LiDAR labels

Use labelCloud to annotate or correct a subset of sequences. The labels stored in each bag directory use the labelCloud `kitti_untransformed` style:

```text
person 0 0 0 0 0 0 0 h w l x y z yaw
```

The currently selected training sequences are defined in `data_processing/prepare_openpcdet_custom.py`.

#### 2. Convert labels for OpenPCDet

`prepare_openpcdet_custom.py` converts labelCloud labels into the OpenPCDet custom format:

```text
x y z dx dy dz heading Pedestrian
```

It also converts each `.bin` point cloud into `.npy` and creates the OpenPCDet `ImageSets` split files.

```bash
$ python3 data_processing/prepare_openpcdet_custom.py --overwrite
```

#### 3. Train the LiDAR detector

OpenPCDet is kept as a local external dependency and is not tracked in this repository. After setting up OpenPCDet, generate dataset infos and train the pedestrian detector:

```bash
$ cd OpenPCDet
$ python3 -m pcdet.datasets.custom.custom_dataset create_custom_infos \
    tools/cfgs/dataset_configs/avped_pedestrian_dataset.yaml
$ cd tools
$ python3 train.py \
    --cfg_file cfgs/custom_models/avped_second_pedestrian.yaml \
    --batch_size 4 \
    --epochs 80 \
    --workers 4 \
    --extra_tag avped_pedestrian
```

#### 4. Infer pseudo labels for the remaining sequences

Use the trained checkpoint to infer bounding boxes for sequences that are not included in the manually checked training set:

```bash
$ cd OpenPCDet/tools
$ python3 infer_avped_pairs.py \
    --cfg_file cfgs/custom_models/avped_second_pedestrian.yaml \
    --ckpt ../output/custom_models/avped_second_pedestrian/avped_pedestrian/ckpt/checkpoint_epoch_80.pth \
    --pairs_root ../../data/pairs \
    --score_thresh 0.1
```

The inference script writes:

```text
bag_name/
  labels/
    0001.txt
  labels_with_scores/
    0001.txt
```

`labels/` keeps the labelCloud-compatible `kitti_untransformed` text format. `labels_with_scores/` appends the detection score as the final field for later filtering.

#### 5. Keep one bbox per frame

For the current single-person setup, if one frame contains multiple predicted boxes, keep only the highest-score box:

```bash
$ python3 data_processing/keep_top_score_label.py
```

#### 6. Check and visualize the generated labels

Check whether every inferred frame contains exactly one box:

```bash
$ python3 data_processing/check_infer_results.py
```

Render only problematic frames:

```bash
$ python3 data_processing/check_infer_results.py --render --only-problems
```

Render all labels for manual review:

```bash
$ python3 data_processing/check_infer_results.py \
    --include-labeled-bags \
    --render \
    --output-dir data/label_previews_all
```

## installation
```bash
$ pip3 install librosa
$ pip3 install open3d
$ pip3 install torchaudio
```

## Tutorial
A tutorial is added to show the general workflow of the network with detailed explanation. Can be found in Tutorial for AVped.ipynb

## Train
```bash
$ python train.py --train_epoch 200 --workers 4 --gpu cuda:0
```
## Evaluate
```bash
$ python evaluation.py --checkpoint_path /model_path --gpu cuda:0 
```

## Note
All the data is collected by ROS system. This [doc](https://docs.google.com/document/d/12u2E4NLQzOtWxfTxPNV5JmIW54Tqq5v7CQbI7BGwuQw/edit?usp=sharing) shows how we process recorded audio data for your reference. 

## Cite
```bash
@inproceedings{yang2023av,
  title={AV-PedAware: Self-Supervised Audio-Visual Fusion for Dynamic Pedestrian Awareness},
  author={Yang, Yizhuo and Yuan, Shenghai and Cao, Muqing and Yang, Jianfei and Xie, Lihua},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1871--1877},
  year={2023},
  organization={IEEE}
}
```
