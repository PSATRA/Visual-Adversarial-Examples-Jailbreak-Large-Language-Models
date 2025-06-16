import os
import argparse
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch.backends.cudnn as cudnn
from gemma3_utils import visual_attacker

import pandas as pd
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma3 visual adversarial attack with APGD")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--instruct_pairs", type=str,
                        default="/gpfs-flash/hulab/zhangwei_srt/lige/princeton/harmful_corpus/JBB-harmful-behaviors.csv")
    parser.add_argument("--clean_img", type=str, 
                        default="/gpfs-flash/hulab/zhangwei_srt/lige/princeton/adversarial_images/clean.jpeg")
    parser.add_argument("--model_ckpt", type=str,
                        default="/gpfs-flash/hulab/zhangwei_srt/lige/weights/gemma-3-4b-it")
    parser.add_argument("--save_dir_adv_img", type=str,
                        default="/gpfs-flash/hulab/zhangwei_srt/lige/princeton/gemma3-4b-unconstrained")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_iter", type=int, default=2000)
    parser.add_argument("--alpha_init", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1.0)   # defualt unconstrained
    parser.add_argument("--rho", type=float, default=0.75)
    args = parser.parse_args()
    return args

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

if __name__ == "__main__":
    args = parse_args()
    
    instruct_id = 0
    seed = args.seed
    instruct_pairs = args.instruct_pairs
    clean_img = args.clean_img
    model_ckpt = args.model_ckpt
    save_dir_adv_img = args.save_dir_adv_img
    device = args.device


    setup_seeds(seed)
    os.makedirs(save_dir_adv_img, exist_ok=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_ckpt, 
        device_map={"": device}, 
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_ckpt)
    model.to(device)
    model.eval()

    # read clean img
    image = Image.open(clean_img).convert("RGB")

    df = pd.read_csv(instruct_pairs)
    sampled_rows = df.sample(n=10)
    instruct_target_pairs = sampled_rows.to_dict(orient="records")
    prompt_text = instruct_target_pairs[instruct_id]["Goal"]
    target_text = instruct_target_pairs[instruct_id]["Target"]
    print("\n=== Text prompt ===")
    print(prompt_text)


    attacker = visual_attacker.APGDAttacker(args=args, model=model)
    attacker.attack(
        processor=processor,
        image=image,
        prompt_text=prompt_text,
        target_text=target_text,
        save_dir=save_dir_adv_img,
        device=device,
    )