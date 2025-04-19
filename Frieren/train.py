import argparse
import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Import the model class dynamically
    components = config['model']['target'].split('.')
    module_name = '.'.join(components[:-1])
    class_name = components[-1]
    
    mod = __import__(module_name, fromlist=[class_name])
    model_class = getattr(mod, class_name)
    
    # Initialize model
    model = model_class(**config['model']['params'])
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=1
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    logger = WandbLogger(project="frame2wav")
    
    # Initialize trainer
    trainer = Trainer(
        gpus=args.gpus,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        max_epochs=100,  # Adjust as needed
        resume_from_checkpoint=args.resume
    )
    
    # Start training
    trainer.fit(model)

if __name__ == "__main__":
    main()
