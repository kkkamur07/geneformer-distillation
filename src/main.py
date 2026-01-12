"""
Main Training Entry Point.

This script orchestrates the entire knowledge distillation training pipeline.
It integrates all components:
- Hydra configuration management.
- Model initialization (Student and Teacher).
- Dataset loading and DataLoader creation.
- DistillationTrainer initialization and execution.

Functions:
    main(cfg): Main function decorated with @hydra.main.

Usage Example:
    ```bash
    # Run with default config
    python src/main.py
    
    # Run with override
    python src/main.py training.batch_size=16
    ```
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from tqdm import tqdm
from transformers.trainer_pt_utils import LengthGroupedSampler

from src.data.collator import GeneDataCollator
from src.model.teacher_model import TeacherModel
from src.model.student_model import StudentModel
from src.data.dataset import GeneformerDataset
from src.training.trainer import DistillationTrainer
from src.training.logging import TrainingLogger
from torch.utils.data import DataLoader
from torch.optim import AdamW


@hydra.main(version_base=None, config_path="/home/krrish/Desktop/Programming/geneformer-scratch/configs", config_name="config")
def main(cfg: DictConfig):
    
    # Print config
    print("=" * 60)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        
        torch.set_float32_matmul_precision('high') 
    # Device
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=cfg.paths.log_dir,
        experiment_name=cfg.names.experiment_name
    )
    
    logger.info("🔧 Loading models...")
    
    # Load teacher model
    teacher = TeacherModel(model_path=cfg.paths.teacher_model_path, device=device)
    logger.info(f"Teacher model loaded from {cfg.paths.teacher_model_path} with {teacher.get_num_parameters():,} parameters")
    
    # Create student model
    student = StudentModel(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        max_position_embeddings=cfg.model.max_position_embeddings,
        hidden_dropout_prob=cfg.model.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.model.attention_probs_dropout_prob,
        device=device
    )
    
    logger.info(f"Student model created with parameters:{student.get_num_parameters():,} total, {student.get_trainable_parameters():,} trainable")
    
    # Compile student model for optimization
    # student = torch.compile(student)
    
    # Load the datasets
    logger.info("Loading datasets...")
    
    train_dataset = GeneformerDataset(cfg.data.dataset_path)
    val_dataset = GeneformerDataset(cfg.data.val_dataset_path)

    # Creating samplers, collators and data loaders
    logger.info("Creating Samplers...")
    
    train_sampler = LengthGroupedSampler(
        batch_size=cfg.training.batch_size,
        dataset=None,
        lengths=train_dataset.length
    )
    
    val_sampler = LengthGroupedSampler(
        batch_size=cfg.training.batch_size,
        lengths=val_dataset.length,     
        dataset=None,            
    )
    
    
    # Creating data collators
    logger.info("Creating DataCollator...")
    
    train_data_collator = GeneDataCollator(
        vocab_size=cfg.model.vocab_size,
        mask_token_id=cfg.model.mask_token_id,
        pad_token_id=cfg.model.pad_token_id,
        mlm_probability=0.15,
    )
    
    # Creating data loaders
    logger.info("Creating DataLoaders...")
    
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=train_data_collator,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        collate_fn=train_data_collator,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None
    )
    
    logger.info(f"DataLoaders created")
    
    # Optimizer
    optimizer = AdamW(
        student.parameters(),
        lr=cfg.training.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        logger=logger
    )
    
    trainer.train()
    
    best_val_loss = trainer.checkpoint_manager.best_val_loss
    
    # Close logger
    logger.close()
    
    print("Training complete!")
    
    return best_val_loss
    


if __name__ == "__main__":
    main()