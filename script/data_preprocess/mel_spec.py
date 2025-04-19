from data_preprocess.NAT_mel import MelNet
import os
from tqdm import tqdm
from glob import glob
import math
import argparse
from argparse import Namespace
import math
import torch
import torchaudio
import numpy as np
from torch.distributed import init_process_group
from torch.utils.data import Dataset,DataLoader,DistributedSampler
import torch.multiprocessing as mp


class audio_dataset(Dataset):
    def __init__(self, root_dir, manifest_path, sr, mode='none', hop_size = None, target_mel_length = None) -> None:
        super().__init__()
        self.root_dir = root_dir
        manifest_file = open(manifest_path, 'r')
        manifest_items = manifest_file.readlines()
        self.audio_paths = [os.path.join(root_dir, item.strip() + '.wav') for item in manifest_items]
        self.sr = sr
        self.mode = mode
        self.target_mel_length = target_mel_length
        self.hop_size = hop_size

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")

        wav, orisr = torchaudio.load(audio_path)
        if wav.shape[0] != 1: # stereo to mono  (2,wav_len) -> (1,wav_len)
            wav = wav.mean(0,keepdim=True)
        wav = torchaudio.functional.resample(wav, orig_freq=orisr, new_freq=self.sr)

        return audio_path,wav
    

def process_audio_by_tsv(rank,args):
    if args.num_gpus > 1:
        init_process_group(backend=args.dist_config['dist_backend'], init_method=args.dist_config['dist_url'],
                            world_size=args.dist_config['world_size'] * args.num_gpus, rank=rank)
    
    sr = args.audio_sample_rate
    dataset = audio_dataset(root_dir=args.root_dir, manifest_path=args.manifest_path, sr=sr, mode=args.mode, hop_size=args.hop_size, target_mel_length=args.batch_max_length)
    sampler = DistributedSampler(dataset, shuffle=False) if args.num_gpus > 1 else None
    # batch_size must == 1,since wav_len is not equal
    loader = DataLoader(dataset, sampler=sampler,batch_size=1, num_workers=16,drop_last=False)

    device = torch.device('cuda:{:d}'.format(rank))
    mel_net = MelNet(args.__dict__)
    mel_net.to(device)

    root = args.save_path
    loader = tqdm(loader) if rank == 0 else loader
    for batch in loader:
        audio_paths,wavs = batch
        wavs = wavs.to(device)

        if args.save_mel:
            mode = args.mode
            batch_max_length = args.batch_max_length
            mel_root = root
            os.makedirs(mel_root,exist_ok=True)

            for audio_path,wav in zip(audio_paths,wavs):
                psplits = audio_path.split('/')
                wav_name = psplits[-1]
                mel_name = wav_name[:-4]+'_mel.npy'
                
                mel_path = os.path.join(mel_root, mel_name)
                if not os.path.exists(mel_path):
                    mel_spec = mel_net(wav).cpu().numpy().squeeze(0) # (mel_bins,mel_len) 
                    if mel_spec.shape[1] <= batch_max_length:
                        if mode == 'tile': # pad is done in dataset as pad wav
                            n_repeat = math.ceil((batch_max_length + 1) / mel_spec.shape[1])
                            mel_spec = np.tile(mel_spec,reps=(1,n_repeat))
                        elif mode == 'none' or mode == 'pad':
                            pass
                        else:
                            raise ValueError(f'mode:{mode} is not supported')
                    mel_spec = mel_spec[:,:batch_max_length]
                    np.save(mel_path,mel_spec)      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=str)
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str)

    pargs = parser.parse_args()

    os.makedirs(pargs.output_dir, exist_ok=True)

    num_gpus = 1
    max_duration = 10
    batch_max_length = int(max_duration * 62.5)# 62.5 is the mel length for 1 second
    vqgan_bathch_max_length = 848 # spec_crop_len in SpecVQGAN.
    vqgan_spec_len = 860

    args = {
        'audio_sample_rate': 16000,
        'audio_num_mel_bins':80,
        'fft_size': 1024,
        'win_size': 1024,
        'hop_size': 256,
        'fmin': 0,
        'fmax': 8000,
        'batch_max_length': batch_max_length, 
        'num_gpus': num_gpus,
        'mode': 'none', # pad,none,
        'save_resample':False,
        'save_mel' :True,
        'root_dir': pargs.input_dir,
        'manifest_path': pargs.manifest_path,
        'save_path': pargs.output_dir,
    }

    vqgan_args = {
        'audio_sample_rate': 22050,
        'audio_num_mel_bins':80,
        'fft_size': 1024,
        'win_size': 1024,
        'hop_size': 256,
        'fmin': 0,
        'fmax': 8000,
        'batch_max_length': batch_max_length, 
        'num_gpus': num_gpus,
        'mode': 'none', # pad,none,
        'save_resample':False,
        'save_mel' :True,
        'root_dir': pargs.input_dir,
        'manifest_path': pargs.manifest_path,
        'save_path': pargs.output_dir,
    }

    args = Namespace(**args)  
    args.dist_config = {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54189",
        "world_size": 1
    }

    if args.num_gpus > 1:
        mp.spawn(process_audio_by_tsv, nprocs=num_gpus, args=(args,))
    else:
        process_audio_by_tsv(0, args=args)
    
    print("done")
    