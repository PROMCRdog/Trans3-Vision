# Dependencies
# #%cd /content

# #!git clone https://github.com/IDEA-Research/Grounded-Segment-Anything

# #%cd Grounded-Segment-Anything 
# !pip install -q -r requirements.txt
# %cd /home/MichaelWang/Grounded-Segment-Anything/GroundingDINO
# !pip install -q .
# %cd /home/MichaelWang/Grounded-Segment-Anything/segment_anything
# !pip install -q .
# %cd /home/MichaelWang/Grounded-Segment-Anything

# Imports
import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import argparse
import copy

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline


from huggingface_hub import hf_hub_download


# #Initial Variables
# my_img_path = "/home/MichaelWang/Inpaint_Imgs/conference50"
# my_img_name = "conference"
# anotation_prompt = "wall"
# inpainting_prompt = ""
# inpainting_neg_prompt = ""


# Load Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Grounding Dino Model
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"


groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)


# SAM
#! wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

sam_checkpoint = 'sam_vit_h_4b8939.pth'

sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))


# Stable Diffusion (Inpainting)
sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to(device)


# Inference
# Load image 
def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(BytesIO(r.content)) as im:
        im.save(image_file_path)
    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))


local_image_path = "assets/inpaint_demo.jpg"
image_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"

#download_image(image_url, local_image_path)
image_source, image = load_image(local_image_path)
Image.fromarray(image_source)


# Grounding Dino for Detection
# detect object using grounding DINO
def detect(image, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
  boxes, logits, phrases = predict(
      model=model, 
      image=image, 
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )

  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
  return annotated_frame, boxes 

#execute
annotated_frame, detected_boxes = detect(image, text_prompt="bench", model=groundingdino_model)
Image.fromarray(annotated_frame)


#SAM for Segmentation
#defs
def segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()
  

def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

#executes
segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)
annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)
Image.fromarray(annotated_frame_with_mask)


#Stable Diffusion for Inpainting
# create mask images 
mask = segmented_frame_masks[0][0].cpu().numpy()
inverted_mask = ((1 - mask) * 255).astype(np.uint8)


image_source_pil = Image.fromarray(image_source)
image_mask_pil = Image.fromarray(mask)
inverted_image_mask_pil = Image.fromarray(inverted_mask)


display(*[image_source_pil, image_mask_pil, inverted_image_mask_pil])


def generate_image(image, mask, prompt, negative_prompt, pipe): #(image, mask, prompt, negative_prompt, pipe, seed):
  # resize for inpainting 
  w, h = image.size
  in_image = image.resize((512, 512))
  in_mask = mask.resize((512, 512))

  #generator = torch.Generator(device).manual_seed(seed) 

  result = pipe(image=in_image, mask_image=in_mask, prompt=prompt, negative_prompt=negative_prompt) #, generator=generator
  result = result.images[0]

  return result.resize((w, h))

#Examples
prompt="a glass table"
negative_prompt="low resolution, ugly, patterns, design, artistic"
#seed = 32 # for reproducibility 

#generated_image = generate_image(image=image_source_pil, mask=image_mask_pil, prompt=prompt, negative_prompt=negative_prompt, pipe=sd_pipe, seed=seed)
#generated_image

# prompt="a beach with turquoise water, sand, and coconuts"
# negative_prompt="people, low resolution, ugly"
# seed = 32 # for reproducibility 

# generated_image = generate_image(image_source_pil, inverted_image_mask_pil, prompt, negative_prompt, sd_pipe, seed)
# generated_image


#My Loop
images_directory = "/home/MichaelWang/Inpaint_Imgs/conference2"

image_files = [file for file in os.listdir(images_directory) if file.endswith((".jpg", ".png", ".jpeg"))]

trans10kv2pallete = [
    0, 0, 0,
    120, 120, 70,
    235, 255, 7,
    6, 230, 230,
    204, 255, 4,
    120, 120, 120,
    140, 140, 140,
    255, 51, 7,
    224, 5, 255,
    204, 5, 255,
    150, 5, 61,
    4, 250, 7]


file_count = 1
for image_file in image_files:
    image_path = os.path.join(images_directory, image_file)
    
    image_source, image = load_image(image_path)

    annotated_frame, detected_boxes = detect(image, text_prompt="sidewall", model=groundingdino_model)
    
    # # Check if anything is detected
    # if (detected_boxes == 0).all:
    #    continue

    # SAM
    try: 
        segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)
        annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)

        # create mask images 
        mask = segmented_frame_masks[0][0].cpu().numpy()
        inverted_mask = ((1 - mask) * 255).astype(np.uint8)

        image_source_pil = Image.fromarray(image_source)
        image_mask_pil = Image.fromarray(mask) # check line 278
        inverted_image_mask_pil = Image.fromarray(inverted_mask)

        # Inpainting (Stable Diffusion)
        prompt = "glass wall inside a modern office building"
        negative_prompt = "low resolution, ugly, patterns, design, artistic"
        #seed = 32  # for reproducibility
        generated_image = generate_image(image=image_source_pil, mask=image_mask_pil, prompt=prompt,
                                        negative_prompt=negative_prompt, pipe=sd_pipe)

        # Print file name
        print(image_file)

        # save the output images
        output_dir = "/home/MichaelWang/Inpaint_Imgs/outputs/conference/converted1"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # os.mkdir(output_dir)
        # os.mkdir(os.path.join(output_dir, "/original"))
        # os.mkdir(os.path.join(output_dir, "/mask"))
        # os.mkdir(os.path.join(output_dir, "/inpainted"))

        # need to convert to PIL
        annotated_frame_with_mask_pil = Image.fromarray(annotated_frame_with_mask)
        image_source_pil = Image.fromarray(image_source)

        # convert image to select color pallet
        image_mask_converted = image_mask_pil.convert("L")
        for i in range(0, image_mask_converted.size[0]):
            for j in range(0, image_mask_converted.size[1]):
                if image_mask_converted.getpixel((i, j)) == 255:
                    image_mask_converted.putpixel(((i, j)), 8) #Change this accordingly to the type generated
        image_mask_converted.putpalette(trans10kv2pallete)

        file_count += 1
        # save
        image_source_pil.save(os.path.join(output_dir, "original/", f"{file_count}.png")) #_original
        image_mask_converted.save(os.path.join(output_dir, "mask/", f"{file_count}_mask.png"))
        generated_image.save(os.path.join(output_dir, "inpainted/", f"{file_count}.jpg")) #_inpainted

    except Exception as e:
        print("my error:", e)
