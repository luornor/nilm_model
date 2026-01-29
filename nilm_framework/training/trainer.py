"""
Unified training loop for NILM models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import numpy as np

from ..config import TrainingConfig
from .metrics import compute_metrics, MetricsTracker


class Trainer:
    """
    Unified trainer for NILM models.
    
    Handles training, validation, and model saving.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            config: Training configuration
            device: Device to use (auto-detect if None)
        """
        self.model = model
        self.config = config
        self.device = device or self._get_device()
        self.model.to(self.device)
        
        # Loss function
        self.criterion = self._create_loss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler (optional)
        self.scheduler = self._create_scheduler()
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker()
    
    def _get_device(self) -> torch.device:
        """Get device (cuda if available, else cpu)."""
        if self.config.device:
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_loss(self) -> nn.Module:
        """Create loss function."""
        if self.config.loss_type == "BCEWithLogitsLoss":
            # Pos weight will be set dynamically if pos_weight_auto is True
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.config.loss_type}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_kwargs = self.config.optimizer_kwargs.copy()
        
        if self.config.optimizer == "Adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                **optimizer_kwargs,
            )
        elif self.config.optimizer == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                **optimizer_kwargs,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler if configured."""
        scheduler_type = getattr(self.config, 'scheduler', None)
        if scheduler_type is None:
            return None
        
        if scheduler_type == "ReduceLROnPlateau":
            patience = getattr(self.config, 'scheduler_patience', 3)
            factor = getattr(self.config, 'scheduler_factor', 0.5)
            # Note: verbose parameter not included for compatibility with older PyTorch versions
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize F1 score
                factor=factor,
                patience=patience,
                verbose=True,
            )
        elif scheduler_type == "StepLR":
            step_size = getattr(self.config, 'scheduler_step_size', 10)
            gamma = getattr(self.config, 'scheduler_gamma', 0.1)
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        else:
            return None
    
    def _compute_pos_weight(self, y: torch.Tensor) -> torch.Tensor:
        """Compute positive weight for balanced loss."""
        pos = float(y.sum())
        neg = float(len(y) - pos)
        if pos == 0:
            return torch.tensor([1.0], device=self.device)
        return torch.tensor([neg / (pos + 1e-6)], device=self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        tracker = MetricsTracker()
        
        all_pred = []
        all_true = []
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Update pos_weight if auto-balancing
            if self.config.pos_weight_auto and isinstance(self.criterion, nn.BCEWithLogitsLoss):
                pos_weight = self._compute_pos_weight(y)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                preds = (probs >= 0.5).astype(int)
                all_pred.extend(preds)
                all_true.extend(y.cpu().numpy().reshape(-1).astype(int))
            
            tracker.update(loss.item())
            
            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Compute epoch metrics
        metrics = compute_metrics(
            np.array(all_true),
            np.array(all_pred),
        )
        metrics["loss"] = tracker.get_mean_loss()
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        tracker = MetricsTracker()
        
        all_pred = []
        all_true = []
        all_prob = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                logits = self.model(x)
                loss = self.criterion(logits, y)
                
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                preds = (probs >= 0.5).astype(int)
                
                all_pred.extend(preds)
                all_true.extend(y.cpu().numpy().reshape(-1).astype(int))
                all_prob.extend(probs)
                
                tracker.update(loss.item())
        
        # Compute metrics
        metrics = compute_metrics(
            np.array(all_true),
            np.array(all_pred),
            y_prob=np.array(all_prob),
        )
        metrics["loss"] = tracker.get_mean_loss()
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            save_dir: Directory to save model checkpoints
            
        Returns:
            Dictionary with training history
        """
        history = {
            "train_metrics": [],
            "val_metrics": [],
        }
        
        best_val_f1 = -1.0
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"Epoch {epoch}/{self.config.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history["train_metrics"].append(train_metrics)
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, "
                  f"Precision: {train_metrics['precision']:.4f}, "
                  f"Recall: {train_metrics['recall']:.4f}")
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history["val_metrics"].append(val_metrics)
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}, "
                      f"Precision: {val_metrics['precision']:.4f}, "
                      f"Recall: {val_metrics['recall']:.4f}")
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["f1"])
                    else:
                        self.scheduler.step()
                
                # Save best model
                if self.config.save_best and val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    if save_dir:
                        self.save_model(os.path.join(save_dir, "best_model.pt"))
            else:
                # Update scheduler even without validation
                if self.scheduler is not None and not isinstance(
                    self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()
                
                # Save after each epoch if no validation
                if save_dir:
                    self.save_model(os.path.join(save_dir, f"epoch_{epoch}.pt"))
        
        # Save final model
        if save_dir:
            self.save_model(os.path.join(save_dir, "final_model.pt"))
        
        return history
    
    def save_model(self, path: str):
        """Save model state dict."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to: {path}")
    
    def load_model(self, path: str):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded model from: {path}")
