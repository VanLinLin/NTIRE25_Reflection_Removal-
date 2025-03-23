# [TEAM ACVLAB][NTIRE 2025 Image Reflection Removal Challenge](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

## Link to the codes/executables of the solution(s):
* [Checkpoints](https://drive.google.com/file/d/1aivtTxrhFUcepZtnYmMupaiEoH27r91o/view?usp=drive_link)
* Input / Output file

## Environments
```bash
conda create -n ntire_reflection python=3.9 -y

conda activate ntire_reflection

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

```

## Folder Structure
```bash
test_dir
├── Origin          <- Put the reflection affected images in this folder
│   ├── test_0000.png
│   ├── test_0001.png
│   ├── ...
├── Depth
├── Normal


output_dir
├── test_0000.png
├── test_0001.png
├──...
```

## How to test?
1. Clone [Depth anything v2](https://github.com/DepthAnything/Depth-Anything-V2.git)

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```
2. Download the [pretrain model of depth anything v2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)

3. Run ```get_depth_normap.py``` to create depth and normal map.
```python
python get_depth_normap.py
```

Now folder structure will be
```bash
test_dir
├── Origin
│   ├── test_0000.png
│   ├── test_0001.png
│   ├── ...
├── Depth
│   ├── test_0000.npy
│   ├── test_0001.npy
│   ├── ...
├── Normal
│   ├── test_0000.npy
│   ├── test_0001.npy
│   ├── ...

output_dir
├── 0000.png
├── 0001.png
├──...
```

1. Clone [DINOv2](https://github.com/facebookresearch/dinov2.git)
```bash
git clone https://github.com/facebookresearch/dinov2.git
```

1. Download [Reflection removal weight](https://drive.google.com/file/d/1aivtTxrhFUcepZtnYmMupaiEoH27r91o/view?usp=drive_link)

```bash 
gdown 1aivtTxrhFUcepZtnYmMupaiEoH27r91o
```

6. Run ```run_test.sh``` to get inference results.

```bash
bash run_test.sh
```
## License and Acknowledgement
This code repository is release under [MIT License](https://github.com/VanLinLin/NTIRE25_reflection_Removal-?tab=MIT-1-ov-file#readme).