from diffusers import StableDiffusionPipeline
import torch
import json
from accelerate import Accelerator, PartialState 
from tqdm import tqdm
import os
from fire import Fire

def inference(ckpt_path, output_dir, metadata_path="data/coco_val/metadata.jsonl", num_processes=1):
    pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16,safety_checker=None)
    distributed_state = PartialState()
    pipe.to(distributed_state.device)
    pipe.set_progress_bar_config(disable=True)
    os.makedirs(output_dir, exist_ok=True)
    with open(metadata_path, "r") as f:
        metadata = [json.loads(line) for line in f.readlines()]
    print(f"Total number of images-caption: {len(metadata)}")
    # only leave the first caption for each image (remove duplicates)
    image_ids = set()
    filtered_metadata = []
    for item in metadata:
        if item["file_name"] not in image_ids:
            image_ids.add(item["file_name"])
            filtered_metadata.append(item)
    metadata = filtered_metadata
    print(f"Total number of images: {len(metadata)}")
    num_processes = distributed_state.num_processes
    if len(metadata) % num_processes != 0:
        metadata += [metadata[-1]] * (num_processes - len(metadata) % num_processes)
    batched_metadata = [metadata[i:i+num_processes] for i in range(0, len(metadata), num_processes)]
    for batch in tqdm(batched_metadata):
        with distributed_state.split_between_processes(batch) as item:
            item = item[0]
            image = pipe(
                prompt = item["Prompt"],
                num_inference_steps = 50,
                guidance_scale = 3.0,
            ).images[0]
            file_name = item["file_name"]
            base_name = os.path.basename(file_name)
            image.save(os.path.join(output_dir, base_name))
            
if __name__ == "__main__":
    Fire(inference)