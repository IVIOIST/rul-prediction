"""
Training utilities for LSTM RUL Prediction
Author: Tony
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm


class LSTMTrainer:
    """Trainer class for LSTM RUL prediction model"""
    
    def __init__(self, model, device, config):
        """
        Args:
            model: LSTM model instance
            device: torch device (cuda or cpu)
            config: configuration object
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,  # Reduce LR by half
            patience=3,  # Wait 3 epochs (reduced from 5 for faster adjustment)
            verbose=True,
            min_lr=1e-6  # Minimum learning rate
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        actuals = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch_x.size(0)
            predictions.extend(outputs.detach().cpu().numpy())
            actuals.extend(batch_y.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader.dataset)
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        
        return avg_loss, rmse
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_x.size(0)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader.dataset)
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        
        return avg_loss, rmse
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: number of epochs (if None, use config)
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        print("=" * 60)
        print(f"Starting Training on {self.device}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_rmse = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_rmse = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['learning_rates'].append(current_lr)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.2f}s) - "
                  f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # Early stopping check
            if val_loss < self.best_val_loss - self.config.MIN_DELTA:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                print(f"  → New best model! (Val Loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"Training completed in {total_time / 60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Restored best model weights")
    
    def save_model(self, path=None):
        """Save model checkpoint"""
        if path is None:
            path = Path(self.config.MODEL_SAVE_PATH) / self.config.BEST_MODEL_NAME
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, path)
        print(f"\nModel saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Model loaded from {path}")
    
    def get_history(self):
        """Get training history"""
        return self.history
