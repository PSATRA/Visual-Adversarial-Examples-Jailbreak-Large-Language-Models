import argparse
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import random
import numpy as np
import torch.backends.cudnn as cudnn
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--instruct_pairs", type=str,
                        default="/gpfs-flash/hulab/zhangwei_srt/lige/princeton/harmful_corpus/JBB-harmful-behaviors.csv")
    parser.add_argument("--adv_img_path", type=str,
                        default="/gpfs-flash/hulab/zhangwei_srt/lige/princeton/gemma3-4b-unconstrained/adv_iter_2000.png")
    parser.add_argument("--model_ckpt", type=str,
                        default="/gpfs-flash/hulab/zhangwei_srt/lige/weights/gemma-3-4b-it")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def generate_responses(model, processor, image, prompt, device, num_samples: int = 10):
    image = image.convert("RGB")
    msg = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    inputs = processor.apply_chat_template(
        msg,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[-1]

    responses = []
    for i in range(num_samples):
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
            )
        gen_ids = gen_out[0][input_len:].unsqueeze(0)
        decoded = processor.decode(gen_ids[0], skip_special_tokens=True)
        print(f"\n=== Response iter {i+1}/{num_samples} ===")
        print(decoded)

if __name__ == "__main__":
    args = parse_args()
    instruct_id = 0
    num_samples = 10
    
    setup_seeds(args.seed)

    device = args.device
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_ckpt,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_ckpt)
    model.to(device)
    model.eval()
    
    instruct_pairs = args.instruct_pairs
    df = pd.read_csv(instruct_pairs)
    sampled_rows = df.sample(n=10)
    instruct_target_pairs = sampled_rows.to_dict(orient="records")
    prompt_text = instruct_target_pairs[instruct_id]["Goal"]
    target_text = instruct_target_pairs[instruct_id]["Target"]
    print("\n=== Text prompt ===")
    print(prompt_text)

    adv_img = Image.open(args.adv_img_path)

    print("\n=== Response on adversarial image ===")
    generate_responses(model, processor, adv_img, prompt_text, device, num_samples)