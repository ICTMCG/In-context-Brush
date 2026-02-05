import torch
from torchvision.ops import box_convert

from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import SamPredictor, sam_model_registry
from pathlib import Path
import json
import os


obj_path = "images/barn.jpg"


bg_path = "images/snow.jpg"
json_path = "input_jsons/snow.json"


obj_subject = "house"

box_treshold = 0.35
text_treshold = 0.25

obj_name = Path(obj_path).stem 
bg_name = Path(bg_path).stem

# Groundingdino
dino_model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                        "model/groundingdino_swint_ogc.pth")
dino_model = dino_model.to('cuda:0')
OBJ_IMAGE_PATH = obj_path
BG_IMAGE_PATH = bg_path
OBJ_TEXT_PROMPT = f"{obj_subject} ."
BOX_TRESHOLD = box_treshold
TEXT_TRESHOLD = text_treshold

obj_image_source, obj_image = load_image(OBJ_IMAGE_PATH)
bg_image_source, bg_image = load_image(BG_IMAGE_PATH)

obj_boxes, obj_logits, obj_phrases = predict(
    model=dino_model,
    image=obj_image,
    caption=OBJ_TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)



# SAM
sam = sam_model_registry["default"](checkpoint="/model/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image(obj_image_source)
h, w, _ = obj_image_source.shape
obj_boxes = obj_boxes * torch.Tensor([w, h, w, h])
obj_boxes = box_convert(boxes=obj_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
obj_masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=obj_boxes[0][None, :],
    multimask_output=False,
)


obj_mask = obj_masks[0]
obj_mask = np.stack([obj_mask, obj_mask, obj_mask], axis=-1)
obj_masked_image = obj_image_source * obj_mask
obj_mask_pil = Image.fromarray((obj_mask * 255).astype('uint8'), 'RGB')

obj_inverse_mask = Image.eval(obj_mask_pil, lambda x: 255 - x)
obj_result = obj_masked_image + np.array(obj_inverse_mask)


img_w, img_h = bg_image_source.shape[0], bg_image_source.shape[1]


image_array = np.array(bg_image_source)
image_array_1 = image_array.copy()

#================================================================================================
with open(json_path, 'r') as f:
    json_data = json.load(f)
first_key = list(json_data.keys())[0]
image_info = json_data[first_key]
regions = image_info["regions"]

output_image = Image.new('RGB', (int(img_w), int(img_h)), (0, 0, 0))

draw = ImageDraw.Draw(output_image)

for region in regions:
    shape = region["shape_attributes"]
    
    if shape["name"] == "ellipse":
        cx = int(shape["cx"])
        cy = int(shape["cy"])
        rx = int(shape["rx"])
        ry = int(shape["ry"])

        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        draw.ellipse(bbox, fill=(255, 255, 255))
    if shape["name"] == "polygon":
        points = [(shape["all_points_x"][i], shape["all_points_y"][i]) for i in range(len(shape["all_points_x"]))]
        draw.polygon(points, fill="white")
output_image.save(f"./output/{shape['name']}_mask.png")


mask = output_image
image_array_2 = np.array(mask)




# paste patches
mask_zero = np.ones_like(obj_image_source) * 0
mask_zero = Image.fromarray(mask_zero).resize((768,768))
image_array_2 = Image.fromarray(image_array_2).resize((768,768))
obj_result = Image.fromarray(obj_result).resize((768,768))
image_array_1 = Image.fromarray(image_array_1).resize((768,768))

mask_diptych = np.concatenate([mask_zero, image_array_2], axis=-3)
image_diptych = np.concatenate([obj_result, image_array_1], axis=-3)


# Load image and mask
size = (768, 768)
image = Image.fromarray(image_diptych)
mask = Image.fromarray(mask_diptych)
image = image.resize((768, 768*2))
mask = mask.resize((768, 768*2))
image.save('./output/image_diptych.png')
mask.save('./output/mask_diptych.png')




pipe = FluxFillPipeline.from_pretrained("model/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU


prompt_list = [
                "A barn",
                "A barn in the snow",
                ]

for prompt in prompt_list:
    image_output = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=768*2,
        width=768,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]

    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{bg_name}_{obj_name}_{prompt[:70]}.png")
    image_output.save(save_path)