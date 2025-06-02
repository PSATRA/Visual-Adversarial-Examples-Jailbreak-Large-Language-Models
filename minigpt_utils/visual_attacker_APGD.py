import torch
import math
from tqdm import tqdm
import random
from minigpt_utils import prompt_wrapper, generator
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import seaborn as sns




def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device)
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device)
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class Attacker:

    def __init__(self, args, model, targets, device='cuda:0', is_rtp=False):

        self.args = args
        self.model = model
        self.device = device
        self.is_rtp = is_rtp

        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []
        
        # APGD
        self.rho = 0.75 # threshold for cond1
        self.init_alpha = None
        self.f_best = -float('inf')
        self.x_best = None

        # APGD checkpoint
        self.checkpoints = []
        self.last_ckpt_idx = 0
        self.since_ckpt_successes = []
        self.since_ckpt_improved = False
        self.prev_loss = None

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)
        
    def make_checkpoints(self, num_iter):
        ps = [0.0, 0.22]   # paper default setting
        while ps[-1] < 1.0:
            pj, pi = ps[-1], ps[-2]
            step = max(pj-pi-0.03, 0.06)
            ps.append(min(pj+step, 1.0))
            
        self.checkpoints = sorted({0} | {math.ceil(p*num_iter) for p in ps})
        
        self.f_best = float('inf')
        self.x_best = None
        self.last_ckpt_idx = 0
        self.since_ckpt_successes.clear()
        self.since_ckpt_improved = False
        self.prev_loss = None
        
    def update_alpha(self, alpha, curr_iter, curr_loss):
        if self.prev_loss is not None: 
            self.since_ckpt_successes.append(curr_loss < self.prev_loss)
        self.prev_loss = curr_loss
        
        if curr_loss < self.f_best:
            self.f_best = curr_loss
            self.since_ckpt_improved = True
        
        if ((self.last_ckpt_idx + 1) < len(self.checkpoints)) and curr_iter == self.checkpoints[self.last_ckpt_idx+1]:  # reach the next ckpt
            success_rate = sum(self.since_ckpt_successes) / len(self.since_ckpt_successes) if self.since_ckpt_successes else 1.0
            
            if success_rate < self.rho or not self.since_ckpt_improved: # cond1 or cond2
                alpha /= 2
            
            self.last_ckpt_idx += 1
            self.since_ckpt_successes.clear()
            self.since_ckpt_improved = False
            
        return alpha

    def attack_unconstrained(self, prompts_train, img, batch_size = 8, num_iter=2000, alpha_init=128):

        print('>>> batch_size:', batch_size)

        my_generator = generator.Generator(model=self.model)
        
        #APGD init
        self.make_checkpoints(num_iter=num_iter)
        alpha = alpha_init
        momentum = 0.75
        prev_noise = None

        adv_noise = torch.rand_like(img).to(self.device) # [0,1]
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        for t in tqdm(range(num_iter + 1)):

            batch_targets = random.sample(self.targets, batch_size)
            
            text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt
            text_prompt = text_prompt_template % prompts_train[num_iter%40]
            text_prompts = [text_prompt] * batch_size


            x_adv = normalize(adv_noise)

            prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
            prompt.img_embs = prompt.img_embs * batch_size
            prompt.update_context_embs()

            target_loss = self.attack_loss(prompt, batch_targets)
            target_loss.backward()
            
            # update state
            if target_loss.item() < self.f_best:
                self.f_best = target_loss
                self.x_best = adv_noise.data.clone()
            
            # APGD
            grad_sign = adv_noise.grad.detach().sign()
            z = (adv_noise + alpha * grad_sign).clamp(0,1)
            
            # momentum
            if prev_noise is None:
                new_noise = z
            else: 
                delta1 = z - adv_noise
                delta2 = adv_noise - prev_noise
                new_noise = (adv_noise - alpha * momentum * delta1 - alpha * (1-momentum) * delta2).clamp(0,1)
            prev_noise = adv_noise.data.clone()
            adv_noise.data.copy_(new_noise)
                
            alpha_new = self.update_alpha(alpha, curr_iter=t+1, curr_loss=target_loss.item())
            if alpha_new < alpha:
                adv_noise.data.copy_(self.x_best)
            alpha = alpha_new

            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            print("target_loss: %f" % (
                target_loss.item())
                  )

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = normalize(adv_noise)
                prompt.update_img_prompts([[x_adv]])
                prompt.img_embs = prompt.img_embs * batch_size
                prompt.update_context_embs()
                with torch.no_grad():
                    response, _ = my_generator.generate(prompt)
                print('>>>', response)

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

    def attack_constrained(self, prompts_train, img, batch_size = 8, num_iter=2000, alpha=1/255, epsilon = 128/255 ):

        print('>>> batch_size:', batch_size)

        my_generator = generator.Generator(model=self.model)


        adv_noise = torch.rand_like(img).to(self.device) * 2 * epsilon - epsilon
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data

        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()


        for t in tqdm(range(num_iter + 1)):

            batch_targets = random.sample(self.targets, batch_size)

            text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt
            text_prompt = text_prompt_template % prompts_train[num_iter%40]
            text_prompts = [text_prompt] * batch_size

            x_adv = x + adv_noise
            x_adv = normalize(x_adv)

            prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
            prompt.img_embs = prompt.img_embs * batch_size
            prompt.update_context_embs()

            target_loss = self.attack_loss(prompt, batch_targets)
            target_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            print("target_loss: %f" % (
                target_loss.item())
                  )

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)
                prompt.update_img_prompts([[x_adv]])
                prompt.img_embs = prompt.img_embs * batch_size
                prompt.update_context_embs()
                with torch.no_grad():
                    response, _ = my_generator.generate(prompt)
                print('>>>', response)

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

    def plot_loss(self):

        sns.set_theme()
        num_iters = len(self.loss_buffer)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
        plt.clf()

        torch.save(self.loss_buffer, '%s/loss' % (self.args.save_dir))

    def attack_loss(self, prompts, targets):

        context_embs = prompts.context_embs

        if len(context_embs) == 1:
            context_embs = context_embs * len(targets) # expand to fit the batch_size

        assert len(context_embs) == len(targets), f"Unmathced batch size of prompts and targets {len(context_embs)} != {len(targets)}"

        batch_size = len(targets)
        self.model.llama_tokenizer.padding_side = "right"

        to_regress_tokens = self.model.llama_tokenizer(
            targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model.max_txt_len,
            add_special_tokens=False
        ).to(self.device)
        to_regress_embs = self.model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        bos = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.bos_token_id
        bos_embs = self.model.llama_model.model.embed_tokens(bos)

        pad = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.pad_token_id
        pad_embs = self.model.llama_model.model.embed_tokens(pad)


        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.model.llama_tokenizer.pad_token_id, -100
        )


        pos_padding = torch.argmin(T, dim=1) # a simple trick to find the start position of padding

        input_embs = []
        targets_mask = []

        target_tokens_length = []
        context_tokens_length = []
        seq_tokens_length = []

        for i in range(batch_size):

            pos = int(pos_padding[i])
            if T[i][pos] == -100:
                target_length = pos
            else:
                target_length = T.shape[1]

            targets_mask.append(T[i:i+1, :target_length])
            input_embs.append(to_regress_embs[i:i+1, :target_length]) # omit the padding tokens

            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length

            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)

        max_length = max(seq_tokens_length)

        attention_mask = []

        for i in range(batch_size):

            # masked out the context from loss computation
            context_mask =(
                torch.ones([1, context_tokens_length[i] + 1],
                       dtype=torch.long).to(self.device).fill_(-100)  # plus one for bos
            )

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = (
                torch.ones([1, num_to_pad],
                       dtype=torch.long).to(self.device).fill_(-100)
            )

            targets_mask[i] = torch.cat( [context_mask, targets_mask[i], padding_mask], dim=1 )
            input_embs[i] = torch.cat( [bos_embs, context_embs[i], input_embs[i],
                                        pad_embs.repeat(1, num_to_pad, 1)], dim=1 )
            attention_mask.append( torch.LongTensor( [[1]* (1+seq_tokens_length[i]) + [0]*num_to_pad ] ) )

        targets = torch.cat( targets_mask, dim=0 ).to(self.device)
        inputs_embs = torch.cat( input_embs, dim=0 ).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)


        outputs = self.model.llama_model(
                inputs_embeds=inputs_embs,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return loss