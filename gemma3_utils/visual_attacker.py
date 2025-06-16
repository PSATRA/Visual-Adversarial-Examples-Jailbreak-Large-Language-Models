import math
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

    
def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    return images * std + mean

class APGDAttacker:
    def __init__(self, args, model):
        self.model = model
        self.num_iter = args.num_iter
        self.rho = getattr(args, "rho", 0.75)
        self.alpha_init = args.alpha_init
        self.epsilon = args.epsilon

        self.checkpoints = []
        self.last_ckpt_idx = 0
        self.since_ckpt_successes = []
        self.since_ckpt_improved = False
        self.prev_loss = None
        self.f_best = float('inf')
        self.make_checkpoints()

        self.model.eval()
        self.model.requires_grad_(False)

    def make_checkpoints(self):
        ps = [0.0, 0.22]    # paper default setting
        while ps[-1] < 1.0:
            pj, pi = ps[-1], ps[-2]
            step = max(pj - pi - 0.03, 0.06)
            ps.append(min(pj + step, 1.0))
        self.checkpoints = sorted({0} | {math.ceil(p * self.num_iter) for p in ps})
        self.f_best = float('inf')
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

        if (self.last_ckpt_idx + 1 < len(self.checkpoints) and
                curr_iter == self.checkpoints[self.last_ckpt_idx + 1]):
            success_rate = (
                sum(self.since_ckpt_successes) / len(self.since_ckpt_successes)
                if self.since_ckpt_successes else 1.0
            )
            if success_rate < self.rho or not self.since_ckpt_improved:
                alpha = alpha / 2.0

            self.last_ckpt_idx += 1
            self.since_ckpt_successes.clear()
            self.since_ckpt_improved = False

        return alpha
    
    def attack(self, processor, image, prompt_text, target_text, save_dir, device):
        msg = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }]

        input_clean = processor.apply_chat_template(
            msg,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)
        input_ids = input_clean["input_ids"]    # (1, seq_len)
        attention_mask = input_clean["attention_mask"]
        origin_pixel_values = input_clean["pixel_values"].detach().clone()
            
        # APGD attack
        adv_noise = torch.rand_like(origin_pixel_values[:, :, : ,:], device=device).detach()
        adv_noise.requires_grad_()

        alpha = self.alpha_init
        
        image_mean = torch.tensor(processor.image_processor.image_mean, device=device).view(1,3,1,1)
        image_std  = torch.tensor(processor.image_processor.image_std,  device=device).view(1,3,1,1)
        min_norm = (0.0 - image_mean) / image_std
        max_norm = (1.0 - image_mean) / image_std
        
        best_loss = float('inf')
        best_noise = adv_noise.detach().clone()
        
        loss_buffer = []
        
        for t in tqdm(range(1, self.num_iter + 1)):
            raw_norm = normalize(adv_noise)
            
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=raw_norm,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            context_embs = outputs.hidden_states[-1]    # img + text embedding
                
            loss = self.attack_loss(
                context_embs=context_embs,
                target_text=target_text,
                llama_model=self.model.language_model,
                llama_tokenizer=processor.tokenizer,
                device=device
            )
            loss.backward()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_noise = adv_noise.detach().clone()
                self.since_ckpt_improved = True
                
            grad = adv_noise.grad.detach()
            adv_noise.data = (adv_noise + alpha * grad.sign()).clamp(0.0, 1.0)
            
            with torch.no_grad():
                cur_raw_norm = normalize(adv_noise)
                ori_norm = normalize(origin_pixel_values)
                delta = (cur_raw_norm - ori_norm).clamp(min=-self.epsilon, max=self.epsilon)
                proj = (normalize(origin_pixel_values) + delta).clamp(min_norm, max_norm)
                adv_noise.data = denormalize(proj)
                
            alpha = self.update_alpha(alpha=alpha, curr_iter=t, curr_loss=loss.item())
            
            adv_noise.grad.zero_()
            self.model.zero_grad()
            
            loss_buffer.append(loss.item())

            if t % 100 == 0:
                with torch.no_grad():
                    save_tensor = adv_noise.squeeze(0).detach().cpu()   # (3, H, W)
                    save_tensor = (save_tensor * 255.0).clamp(0, 255).byte()
                    arr = save_tensor.permute(1, 2, 0).numpy()
                    pil_img = Image.fromarray(arr)
                    pil_img.save(f"{save_dir}/adv_iter_{t:04d}.png")

                print(
                    f"Iter {t}/{self.num_iter} | "
                    f"curr_loss = {loss.item():.6f} | "
                    f"best_loss = {best_loss:.6f} | "
                )
            
        with torch.no_grad():
            save_tensor = best_noise.squeeze(0).detach().cpu()
            save_tensor = (save_tensor * 255.0).clamp(0, 255).byte()
            arr = save_tensor.permute(1, 2, 0).numpy()
            pil_img = Image.fromarray(arr)
            pil_img.save(f"{save_dir}/adv_best_noise.png")
        plt.figure()
        plt.plot(range(1, len(loss_buffer) + 1), loss_buffer, label='Target Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/loss_curve.png")
        plt.close()
            
    def attack_loss(self, context_embs, target_text, llama_model, llama_tokenizer, device):
        to_regress_tokens = llama_tokenizer(
            target_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False
        ).to(device)

        embed_layer = llama_model.get_input_embeddings()
        to_regress_embs = embed_layer(to_regress_tokens.input_ids)  # shape: (1, target_seq_len, hidden_dim)

        bos_id = torch.tensor([[llama_tokenizer.bos_token_id]], device=device)  # (1,1)
        bos_embs = embed_layer(bos_id)  # (1,1,hidden_dim)

        pad_id = torch.tensor([[llama_tokenizer.pad_token_id]], device=device)
        pad_embs = embed_layer(pad_id)

        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == llama_tokenizer.pad_token_id, -100
        )
        pos_padding = torch.argmin(T, dim=1)  # first -100

        context_length = context_embs.size(1)  # (1, seq_len_ctx, hidden_dim)

        pos = int(pos_padding[0].item())
        if T[0][pos] == -100:
            target_length = pos
        else:
            target_length = T.size(1)

        seq_length = context_length + target_length

        inputs_embs = torch.cat([
            bos_embs,                           # (1, 1, hidden_dim)
            context_embs,                       # (1, context_length, hidden_dim)
            to_regress_embs[:, :target_length], # (1, target_length, hidden_dim)
            pad_embs.repeat(1, max(0, seq_length - (1 + context_length + target_length)), 1)
        ], dim=1)
        total_len = inputs_embs.size(1)

        attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)

        labels = torch.full((1, total_len), -100, dtype=torch.long, device=device)
        start_target = context_length + 1
        labels[0, start_target:start_target + target_length] = to_regress_tokens.input_ids[0, :target_length]

        outputs = llama_model(
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )
        return outputs.loss