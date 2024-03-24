# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch.utils.data import DataLoader
from datetime import datetime
import os
import json
import tyro
from tqdm.auto import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from ...io_util import torch_to_pil
from .dtu import DTUDataset,DTUConfig   



def load_blip2_model(pretrained_model_name_or_path: str = "Salesforce/blip2-opt-2.7b", device: str = "cuda"):
    processor = Blip2Processor.from_pretrained(pretrained_model_name_or_path)
    model = Blip2ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
    model = model.to(device)

    return model, processor


def save_captions_file(captions, output_file: str, intermediate: bool = False):
    date_time = datetime.now().strftime("%d.%m.%Y_%H:%M:%S.%f")
    output_file_parts = output_file.split(".")
    output_file_without_suffix = ".".join(output_file_parts[:-1])
    output_file_without_suffix += f"_{date_time}"
    if intermediate:
        output_file_without_suffix += f"_intermediate"
    output_file_with_time = f"{output_file_without_suffix}.{output_file_parts[-1]}"
    with open(output_file_with_time, "w") as f:
        print(captions)
        json.dump(captions, f, indent=4)


@torch.no_grad()
def generate_blip2_captions(
    dataset_config: DTUConfig,
    pretrained_model_name_or_path: str = "Salesforce/blip2-opt-2.7b",
    device: str = "cuda",
    batch_size: int = 4,
    output_file: str = "co3d_blip2_captions.json",
):
    # load blip2 model
    model, processor = load_blip2_model(pretrained_model_name_or_path, device)

    # make sure the important fields are set correctly
   

    # Get the dataset: parse CO3Dv2
    dataset_config.threshold=0

    # loop over data
    captions = {}
    import json
    captions = json.load(open("/root/autodl-tmp/mvs_training/dtu/dtu_blip2_captions.json"))
    print(captions.keys())
    print("generate captions")
    for split in ["val", "test","train"]:
        dataset_config.split = split
        dataset = DTUDataset(dataset_config)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        for idx, batch in enumerate(tqdm(dataloader, desc="Generate Captions")):
            # get sequence, category from batch
            scans = [x for x in batch["scan_vid"][0]]
            ref_view = [x.item() for x in batch["scan_vid"][1]]
            check = True
            for scan,ref in zip(scans,ref_view):
                if scan not in captions:
                    print(scan)
                    check = False
                    break
                if ref not in captions[scan] and str(ref) not in captions[scan]:
                    print(ref)
                    check = False
                    break
            if check:
                continue
                

            # get image from batch
            try:
                images = batch["images"][:,[0]]  # (batch_size, K=1, C, H, W)
                images = images.squeeze()  # (batch_size, C, H, W)
                images = [torch_to_pil(x) for x in images]  # processor expects PIL images
            except:
                print("Error in images")
                continue

            # run captioning
            inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
            try:
                generated_ids = model.generate(**inputs)
            except Exception as e:
                print(f"Error at idx {idx}: {e}")
                # generate ids of length 1 with random values
                generated_ids = torch.randint(0, 1000, (1, 1), dtype=torch.long, device=device)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_text = [s.strip() for s in generated_text]

            # save captions
            for c, s, p in zip(scans, ref_view, generated_text):

                print(c, s, p, batch["scan_vid"])
                if c not in captions:
                    captions[c] = {}
                if s not in captions[c]:
                    captions[c][s] = []
                captions[c][s].append(p)

            # save intermediate outputs in case this crashes at some point
            if idx % 5000 == 0:
                save_captions_file(captions, output_file, intermediate=True)
        save_captions_file(captions, output_file)

    # save final file
    save_captions_file(captions, output_file)


if __name__ == "__main__":
    tyro.cli(generate_blip2_captions)
