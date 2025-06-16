# 1. Download Vicuna's weights to ./models   (it's a delta version)
# 2. Download LLaMA's weight via: https://huggingface.co/huggyllama/llama-13b/tree/main
# 3. merge them and setup config
# 4. Download the mini-gpt4 compoents' pretrained ckpts
# 5. vision part will be automatically download when launching the model


import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from minigpt_utils import visual_attacker, visual_attacker_APGD, prompt_wrapper

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry



def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=500, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')

    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================


print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print('[Initialization Finished]\n')



if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)



import csv

#target_corpus = "harmful_corpus/derogatory_corpus.csv"
target_corpus = "harmful_corpus/sexism.csv" #TODO

file = open(target_corpus, "r")
data = list(csv.reader(file, delimiter=","))
file.close()
targets = []
num = len(data)
for i in range(num):
    targets.append(data[i][0])
print(targets)


my_attacker = visual_attacker_APGD.Attacker(args, model, targets, device=model.device, is_rtp=False)

template_img = 'adversarial_images/clean.jpeg'
img = Image.open(template_img).convert('RGB')
img = vis_processor(img).unsqueeze(0).to(model.device)


file = open("harmful_corpus/manual_harmful_instructions.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
prompts_train = []
num = len(data)
for i in range(num):
    prompts_train.append(data[0][0])    #TODO
#print("######################")
#print(f"prompt_num: {len(prompts_train)}")
#print(prompts_train)
#print("######################")



if not args.constrained:
    adv_img_prompt = my_attacker.attack_unconstrained(prompts_train,
                                                            img=img, batch_size=4,
                                                            num_iter=args.n_iters, alpha_init=args.alpha/255)

else:
    adv_img_prompt = my_attacker.attack_constrained(prompts_train,
                                                            img=img, batch_size=4,
                                                            num_iter=args.n_iters, alpha_init=args.alpha/255,
                                                            epsilon=args.eps/255)

save_image(adv_img_prompt, '%s/bad_prompt.bmp' % args.save_dir)
print('[Done]')
