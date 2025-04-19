import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

from cfm.train import instantiate_from_config

class VQGANWrapper(pl.LightningModule):
    def __init__(self,
                 vqgan_config,
                 embed_dim=20,
                 image_key="image",
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 learning_rate=4.5e-6
                 ):
        super().__init__()
        self.image_key = image_key
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        
        # Initialize the VQGAN
        self.vqgan = instantiate_from_config(vqgan_config)
        
        # Initialize from checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
        # Get VQGAN's latent dimensions
        self.vqgan_z_channels = self.vqgan.quantize.e_dim
        
        # Adapter layers to handle shape differences
        self.adapter_in = nn.Conv1d(80, 1, 1)  # From 80 channels to 1 channel
        self.adapter_latent = nn.Conv2d(self.vqgan_z_channels, embed_dim, 1)  # From VQGAN latent dim to desired embed_dim
        self.adapter_out = nn.Conv1d(1, 80, 1)  # From 1 channel back to 80 channels
        
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.vqgan.load_state_dict(sd, strict=False)
        print(f"Restored VQGAN from {path}")

    def encode(self, x):
        # x has shape (batch_size, 80, T)
        # Adapt it to VQGAN's expected shape (batch_size, 1, 80, T)
        x_reshaped = self.adapter_in(x)  # (batch_size, 1, T)
        x_reshaped = x_reshaped.permute(0, 2, 1)  # (batch_size, T, 1)
        
        # Reshape to 2D image format expected by VQGAN
        batch_size, seq_len, _ = x_reshaped.shape
        height = 80  # Number of mel bands
        width = max(seq_len // height, 1)  # Calculate width to fit sequence
        padding = width * height - seq_len
        
        if padding > 0:
            # Pad sequence if needed
            x_reshaped = F.pad(x_reshaped, (0, 0, 0, padding))
            
        x_reshaped = x_reshaped.reshape(batch_size, 1, height, width)  # (batch_size, 1, 80, width)
        
        # Encode with VQGAN
        h = self.vqgan.encoder(x_reshaped)
        h = self.vqgan.quant_conv(h)
        quant, _, info = self.vqgan.quantize(h)
        
        # quant has shape (batch_size, vqgan_z_channels, H', W')
        # Adapt to the desired shape
        adapted_quant = self.adapter_latent(quant)
        adapted_quant = adapted_quant.reshape(adapted_quant.shape[0], self.embed_dim, -1)  # (batch_size, embed_dim, H'*W')
        
        # Return a DiagonalGaussianDistribution-like object
        posterior = DiagonalGaussianDistributionAdapter(adapted_quant)
        return posterior

    def decode(self, z):
        # z has shape (batch_size, embed_dim, T')
        # Need to reshape to match VQGAN latent space dimensions
        batch_size, _, latent_seq_len = z.shape
        
        # Estimate VQGAN latent dimensions based on input
        latent_height = 5  # Approximated based on downsampling
        latent_width = max(latent_seq_len // latent_height, 1)
        
        z_reshaped = z.reshape(batch_size, self.embed_dim, latent_height, latent_width)
        
        # Convert back to VQGAN's latent dimensions
        z_adapted = F.conv2d(z_reshaped, self.adapter_latent.weight.transpose(0, 1).unsqueeze(-1).unsqueeze(-1))
        
        # Decode with VQGAN
        z_post = self.vqgan.post_quant_conv(z_adapted)
        x_decoded = self.vqgan.decoder(z_post)
        
        # x_decoded has shape (batch_size, 1, height, width)
        # Reshape back to (batch_size, 80, T)
        batch_size, channels, height, width = x_decoded.shape
        x_flat = x_decoded.reshape(batch_size, channels, height * width)
        x_out = self.adapter_out(x_flat)  # (batch_size, 80, T)
        
        return x_out

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        assert len(x.shape) == 3
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        
        if optimizer_idx == 0:
            # Train encoder+decoder
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                          last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        
        if optimizer_idx == 1:
            # Train discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                              last_layer=self.get_last_layer(), split="train")
            
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                      last_layer=self.get_last_layer(), split="val")
        
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                          last_layer=self.get_last_layer(), split="val")
        
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.adapter_in.parameters()) +
                                 list(self.adapter_latent.parameters()) +
                                 list(self.adapter_out.parameters()),
                                 lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                   lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.adapter_out.weight

    def test_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)  # inputs shape:(b,mel_len,T)
        reconstructions, posterior = self(inputs)  # reconstructions:(b,mel_len,T)
        mse_loss = torch.nn.functional.mse_loss(reconstructions, inputs)
        self.log('test/mse_loss', mse_loss)
          
        test_ckpt_path = os.path.basename(self.trainer.tested_ckpt_path)
        savedir = os.path.join(self.trainer.log_dir, f'output_imgs_{test_ckpt_path}', 'fake_class')
        if batch_idx == 0:
            print(f"save_path is: {savedir}")
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            print(f"save_path is: {savedir}")

        file_names = batch['f_name']
        reconstructions = reconstructions.cpu().numpy()
        for b in range(reconstructions.shape[0]):
            vname_num_split_index = file_names[b].rfind('_')
            v_n, num = file_names[b][:vname_num_split_index], file_names[b][vname_num_split_index+1:]
            save_img_path = os.path.join(savedir, f'{v_n}.npy')
            np.save(save_img_path, reconstructions[b])
        
        return None

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        
        if not only_inputs:
            xrec, posterior = self(x)
            # Generate random samples
            random_latent = torch.randn_like(posterior.sample())
            log["samples"] = self.decode(random_latent).unsqueeze(1)
            log["reconstructions"] = xrec.unsqueeze(1)
        log["inputs"] = x.unsqueeze(1)
        return log


class DiagonalGaussianDistributionAdapter:
    """Adapter class to mimic DiagonalGaussianDistribution behavior needed for AutoencoderKL"""
    
    def __init__(self, parameters):
        self.parameters = parameters
        batch_size, channels, seq_len = parameters.shape
        # Create dummy mean and logvar for KL loss computation
        self.mean = parameters
        self.logvar = torch.zeros_like(parameters)
        
    def sample(self):
        # In VQGAN there's no sampling, just return parameters directly
        return self.parameters
    
    def mode(self):
        return self.parameters
    
    def kl(self):
        # Return a zero KL divergence - no regularization in this case
        return torch.zeros(self.parameters.shape[0], device=self.parameters.device)
