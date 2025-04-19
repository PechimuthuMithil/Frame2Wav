import os
import sys
import glob
import random
import argparse

import torch
import numpy as np
import soundfile as sf

from tqdm import tqdm
from omegaconf import OmegaConf

# More robust path handling
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
frieren_path = os.path.join(project_root, "Frieren")
if frieren_path not in sys.path:
    sys.path.insert(0, frieren_path)

print("Project root:", project_root)
print("Frieren path:", frieren_path)
print("sys.path:", sys.path)

try:
    from cfm.util import instantiate_from_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Checking if module exists:")
    cfm_path = os.path.join(frieren_path, "cfm")
    util_path = os.path.join(cfm_path, "util.py")
    print(f"cfm directory exists: {os.path.exists(cfm_path)}")
    print(f"util.py exists: {os.path.exists(util_path)}")
    if os.path.exists(cfm_path) and not os.path.exists(os.path.join(cfm_path, "__init__.py")):
        print("WARNING: cfm directory exists but has no __init__.py file")
    raise


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def generate_mel(ldm_config_path, ldm_ckpt_path, cavp_root_path, gpus, prefix, timesteps, solver, scale, sample_num):
    # Set Device:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device = torch.device("cuda")    
    seed_everything(21)

    sample_num = sample_num
    cfg_scale = scale    
    steps = timesteps      
    sampler = solver  
    truncate_len = 32

    save_mel_path = os.path.join(prefix, "CFG{}_{}_{}".format(cfg_scale, sampler, steps))
    os.makedirs(save_mel_path, exist_ok=True)

    config = OmegaConf.load(ldm_config_path)
    latent_diffusion_model = load_model_from_config(config, ldm_ckpt_path)

    np_post = '.npy'
    all_npy = sorted(glob.glob(f"{cavp_root_path}/*.npy"))
    
    if len(all_npy)==0:
        all_npy = sorted(glob.glob(f"{cavp_root_path}/*.npz"))
        np_post = '.npz'

    for cavp_path in tqdm(all_npy): 
        last_mel_path = os.path.join(save_mel_path, 'Y' + cavp_path.split('/')[-1].replace(np_post, f'_{sample_num-1}.npy'))
        if os.path.exists(last_mel_path):
            continue

        cavp_feats = np.load(cavp_path)
        if np_post=='.npz':
            cavp_feats = cavp_feats['feat']

        video_feat = torch.Tensor(cavp_feats).unsqueeze(0).repeat(sample_num, 1, 1).to(device)

        feat_len = video_feat.shape[1]
        window_num = feat_len // truncate_len

        mel_list = []   
        for i in tqdm(range(window_num), desc="Window:"):
            start, end = i * truncate_len, (i+1) * truncate_len
            embed_cond_feat = latent_diffusion_model.get_learned_conditioning(video_feat[:, start:end])     

            audio_samples, _ = latent_diffusion_model.sample_param_cfg(embed_cond_feat, cfg_scale=cfg_scale, batch_size=sample_num, solver=sampler, timesteps=steps)
            audio_samples = latent_diffusion_model.decode_first_stage(audio_samples)

            if len(audio_samples.shape) == 4:
                audio_samples = audio_samples[:, 0, :, :].detach().cpu().numpy()                               
            else:
                audio_samples = audio_samples.detach().cpu().numpy()     
            
            mel_list.append(audio_samples)
        
        # Save Samples:
        for i in range(sample_num):     
            mel_path = os.path.join(save_mel_path, 'Y' + cavp_path.split('/')[-1].replace(np_post, f'_{i}.npy'))
            current_mel_list = []
            for k in range(window_num):
                current_mel_list.append(mel_list[k][i])
            if len(current_mel_list) > 0:
                current_mel = np.concatenate(current_mel_list, 1)
                np.save(mel_path, current_mel)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--ldm',
        type=str,
        help='ldm model checkpoints path',
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='ldm model config path',
    )
    parser.add_argument(
        '-g',
        '--gpus',
        type=str,
        default="0",
        help='use gpu',
    )
    parser.add_argument(
        '-f',
        '--cavp_feature',
        type=str,
        help='cavp feature path',
    )
    parser.add_argument(
        '-o',
        '--out-dir',
        type=str,
        help='gen mel save directory',
    )
    parser.add_argument(
        '-t',
        '--timesteps',
        type=int,
        default=26,
    )
    parser.add_argument(
        '-s',
        '--solver',
        type=str,
        default='euler',
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=4.5,
    )
    parser.add_argument(
        '--sample-num',
        type=int,
        default=10
    )
    
    args = parser.parse_args()
    generate_mel(args.config, args.ldm, args.cavp_feature, args.gpus, args.out_dir, args.timesteps, args.solver, args.scale, args.sample_num)
