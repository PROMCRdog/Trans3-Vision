# Trans3-Vision: Transfer learning based Transformer for Transparent Object Segmentation with Grounded-ID2
**Junyang Wang, Zhongshu Liu**

We developed an improved transfer learning-based transformer for transparent objec t segmentation, Trans3-Vision。

We also created a complete pipeline and methodolgy for generating unlimited training image data for any area of semantic segmentation by building upon Stable Diffusion, Grounding Dino, and SAM.


![Grounded-ID2 Process and Design](./assets/Grounded-ID2_Process.png)



## 

## Installation

### Stable Diffusion
For specific details, please reference [web-diffusion]()

### Trans3Vision
Create environment:

```bash
conda create -n trans3-trans python=3.7
conda activate trans4trans
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pyyaml pillow requests tqdm ipython scipy opencv-python thop tabulate
```

And install:

```bash
python setup.py develop --user
```

### Pipeline

### SAM
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

## Grounded-ID2 Dataset
The **Grounded-ID2** (Grounded Integrated Diffusion Inpainting Dataset) process produces high quality images and transparent object masks from input prompts of only text, and it can be used to generate images and pixel-accurate masks for all objects.

<p float="left">
  <img src="assets/Conference_livingroom.jpg" width="45%" />
  <img src="assets/Office_glassdoor.jpg" width="45%" />
</p>
[Dropbox](https://www.dropbox.com/scl/fi/2veeevgbn8z58wbzpiega/Grounded_ID2_only.zip?rlkey=vat92ypn0ehbkurpvfbqbfwoo&dl=0)

## Evaluation Training Weights