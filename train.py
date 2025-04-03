import torch
import os
import time
import math
import logging
import traceback
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from sklearn.metrics import  r2_score, accuracy_score, roc_auc_score, f1_score, mean_absolute_error,confusion_matrix,precision_score,recall_score,mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR, OneCycleLR
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
from rdkit.Chem import Draw
from scipy import stats
import pandas as pd
from data_preprocess import dataload,create_optimized_dataloaders
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import argparse
from sklearn.preprocessing import KBinsDiscretizer
import random
import json
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from DSSA_model import DSSA
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Dict, Any

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

#########################################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description='GTSA Model Training')
    parser.add_argument('-n', '--num-epochs', type=int, default=200)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-p', '--result-path', type=str, default='results')
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('--hp-search', action='store_true', help='Run hyperparameter search')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def subgraph_augment(data, p=0.8):
    num_nodes = data.x.size(0)
    num_keep = int(num_nodes * p)
    keep_nodes = torch.randperm(num_nodes)[:num_keep]
    x = data.x[keep_nodes]
    edge_index = data.edge_index
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    mask = (edge_index[0].unsqueeze(0) == keep_nodes.unsqueeze(1)).any(dim=0) & \
           (edge_index[1].unsqueeze(0) == keep_nodes.unsqueeze(1)).any(dim=0)
    edge_index = edge_index[:, mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    _, edge_index = torch.unique(edge_index, dim=1, return_inverse=True)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=data.y)

class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, pred, target):
        loss = (pred - target)**2
        weight = 1 + self.alpha * torch.abs(target) + self.beta * (target**2)
        return (weight * loss).mean()

class NonNegativeConstraint(nn.Module):
    def forward(self, x):
        return F.softplus(x)
def assign_bin(value, bins):
    return np.digitize(value, bins) - 1

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def augment_data(data):
    if random.random() < 0.4:
        noise = torch.randn_like(data.x) * 0.1
        data.x = data.x + noise
    if random.random() < 0.3:
        edge_mask = torch.rand(data.edge_index.size(1)) > 0.15
        data.edge_index = data.edge_index[:, edge_mask]
    if random.random() < 0.2:
        mask = torch.rand(data.x.size()) > 0.1
        data.x = data.x * mask.float()
    return data

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

##############################################################################################################

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary),
        "recall": recall_score(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "pr_auc": pr_auc,  
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }
    return metrics

def format_metrics(metrics, prefix=""):
    return (f"{prefix} "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"ROC-AUC: {metrics['roc_auc']:.4f}, "
            f"PR-AUC: {metrics['pr_auc']:.4f}, "  
            f"Sensitivity: {metrics['sensitivity']:.4f}, "
            f"Specificity: {metrics['specificity']:.4f}, "
            f"MSE: {metrics['mse']:.4f}, "
            f"MAE: {metrics['mae']:.4f}, "
            f"RMSE: {metrics['rmse']:.4f}, "
            f"R2: {metrics['r2']:.4f}")


#################################################################################


def create_ensemble_model(fold_results, num_node_features, num_edge_features, num_node_types, num_edge_types, device_manager):
    logging.info(f"Starting ensemble model creation with {len(fold_results)} fold results")
    ensemble_models = []
    if not fold_results:
        raise ValueError("Fold results is empty")
    
    for fold_idx, (model_path, val_loss) in enumerate(fold_results):
        try:
            logging.info(f"Processing fold {fold_idx + 1}/{len(fold_results)}")
            logging.info(f"Loading model from path: {model_path}")
            if not os.path.exists(model_path):
                logging.error(f"Model file not found: {model_path}")
                continue
            model = DSSA(
                in_dim=num_node_features,
                hidden_dim=256,
                num_layers=6,
                num_heads=8,
                dropout=0.2,
                num_classes=1,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                processing_steps=4
            ).to(device_manager.device)
      
            try:
                checkpoint = torch.load(model_path)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
                logging.info(f"Successfully loaded state dict for fold {fold_idx + 1}")
            except Exception as e:
                logging.error(f"Error loading state dict for fold {fold_idx + 1}: {str(e)}")
                continue
        
            model = model.to(device_manager.device)
            model.eval()
  
            try:
                test_x = torch.randn(2, num_node_features).to(device_manager.device)
                test_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device_manager.device)
                test_batch = torch.tensor([0, 0], dtype=torch.long).to(device_manager.device)
                with torch.no_grad():
                    test_data = Data(x=test_x, edge_index=test_edge_index, batch=test_batch)
                    test_output = model(test_data)
                if test_output is not None:
                    ensemble_models.append(model)
                    logging.info(f"Successfully verified model for fold {fold_idx + 1}")
                else:
                    logging.error(f"Model verification failed for fold {fold_idx + 1}: null output")
            except Exception as e:
                logging.error(f"Error verifying model for fold {fold_idx + 1}: {str(e)}")
                continue
        except Exception as e:
            logging.error(f"Error processing fold {fold_idx + 1}: {str(e)}")
            continue
    if not ensemble_models:
        raise ValueError("No models could be loaded for ensemble. Check the logs for details.")
    logging.info(f"Successfully created ensemble with {len(ensemble_models)} models")
    return EnsembleModel(ensemble_models)

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        logging.info(f"Initialized EnsembleModel with {len(models)} models")
    def forward(self, data):
        outputs = []
        for idx, model in enumerate(self.models):
            try:
                model.eval()
                with torch.no_grad():
                    output = model(data)      
                if output is not None:
                    outputs.append(output)
                else:
                    logging.warning(f"Model {idx} produced null output")   
            except Exception as e:
                logging.error(f"Error in model {idx} prediction: {str(e)}")
                continue
        if not outputs:
            raise RuntimeError("No valid outputs from any ensemble model")
        stacked_outputs = torch.stack(outputs)
        return torch.mean(stacked_outputs, dim=0)
    
def k_fold_cross_validation(dataset, model_class,num_edge_types, num_node_types,num_node_features, num_edge_features, k=5, num_epochs=500, learning_rate=0.001, patience=50):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset), 1):
        logging.info(f"\nStarting Fold {fold}/{k}")
        train_dataset = [dataset[i] for i in train_indices]
        val_dataset = [dataset[i] for i in val_indices]
        train_loader = PyGDataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=256, shuffle=False)
        save_dir = os.path.join('models', f'run_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f'model_fold_{fold}.pth')
        try:
            model = DSSA(
            in_dim=num_node_features,
            hidden_dim=64,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            num_classes=1,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            processing_steps=4
        ).to(DEVICE)
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
            criterion = nn.BCEWithLogitsLoss()
          
            steps_per_epoch = len(train_loader)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate * 3,
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos'
            )
            best_val_loss = float('inf')
            early_stopping_counter = 0
            best_model_state = None
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Fold {fold}, Epoch {epoch+1}")):
                    batch = batch.to(DEVICE)
                    optimizer.zero_grad()
                    output = model(batch)
                    loss = criterion(output.view(-1), batch.y.float())

                 
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    train_loss += loss.item()
                    if batch_idx % 50 == 0:
                        logging.info(f"Fold {fold}, Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                                   f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                
                avg_train_loss = train_loss / len(train_loader)
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(DEVICE)
                        output = model(batch)
                        loss = criterion(output.view(-1), batch.y.float())
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)       
                logging.info(f"Fold {fold}, Epoch {epoch+1} - "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict()
                    early_stopping_counter = 0
                    logging.info(f"New best validation loss: {best_val_loss:.4f}")
                else:
                    early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            if best_model_state is not None:
                torch.save(best_model_state, model_save_path)
                logging.info(f"Saved best model for fold {fold} to {model_save_path}")
                fold_results.append((model_save_path, best_val_loss))
            else:
                logging.error(f"No best model state found for fold {fold}")
            
        except Exception as e:
            logging.error(f"Error in fold {fold}: {str(e)}")
            traceback.print_exc()
            continue
    if not fold_results:
        raise ValueError("No successful folds completed")
    logging.info(f"Completed {len(fold_results)} successful folds")
    for i, (path, loss) in enumerate(fold_results, 1):
        logging.info(f"Fold {i}: Path = {path}, Best Val Loss = {loss:.4f}")
    
    return fold_results

#############################################################################################################################


class DeviceDataLoader:
    """Wrap a dataloader to move data to device"""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
    def __len__(self):
        return len(self.dataloader)
    def __iter__(self):
        for batch in self.dataloader:
            yield process_batch_to_device(batch, self.device)

class DeviceManager:
    def __init__(self, device=None):
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        print(f"DeviceManager initialized with device: {self.device}")

    def prepare_model(self, model):
        return model.to(self.device)

    def prepare_batch(self, batch):
        return process_batch_to_device(batch, self.device)

    def prepare_optimizer(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        return optimizer

def process_batch_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [b.to(device) if hasattr(b, 'to') else b for b in batch]
    return batch.to(device)
def run_epoch(model, loader, criterion, optimizer=None, device_manager=None):
    device = device_manager.device
    model = device_manager.prepare_model(model)
    model.train() if optimizer else model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch_idx, data in enumerate(tqdm(loader, desc="Epoch Progress")):
        try:
            data = data.to(device)
            with torch.set_grad_enabled(optimizer is not None):
                out = model(data)
                out = out.view(-1)  
                targets = data.y.float().view(-1)  
                if batch_idx == 0:  
                    print(f"Output shape: {out.shape}")
                    print(f"Target shape: {targets.shape}")
                targets = targets.to(device)
                loss = criterion(out, targets)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer = device_manager.prepare_optimizer(optimizer)
                optimizer.step()
            total_loss += loss.item()
            all_preds.extend(out.detach().cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            print(f"Output shape: {out.shape if 'out' in locals() else 'N/A'}")
            print(f"Target shape: {targets.shape if 'targets' in locals() else 'N/A'}")
            continue
    
    if len(all_preds) == 0:
        raise RuntimeError("No predictions were generated ")
    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)

def debug_batch(batch, prefix=""):
    print(f"\n{prefix} Batch Debug Information:")
    print("-" * 50)
    if isinstance(batch, tuple):
        print("✓ Batch contains both graph and SMILES data")
        graph_data, smiles_data = batch
        # print(f"SMILES data present: {len(smiles_data)} sequences")
        # print("\nGraph Data:")
        _debug_graph_data(graph_data)
        # print("\nSMILES Data Sample:")
        print(smiles_data[:3] if len(smiles_data) > 3 else smiles_data)
    else:
        print("✗ Batch contains only graph data (no SMILES)")
        _debug_graph_data(batch)
def _debug_graph_data(data):
    """Helper function to debug graph data"""
    print(f"Data type: {type(data)}")
    if hasattr(data, 'x'):
        print(f"Node features (x):")
        print(f"  Shape: {data.x.shape}")
        print(f"  Device: {data.x.device}")
        print(f"  Type: {data.x.dtype}")
    if hasattr(data, 'edge_index'):
        print(f"Edge index:")
        print(f"  Shape: {data.edge_index.shape}")
        print(f"  Device: {data.edge_index.device}")
    if hasattr(data, 'y'):
        print(f"Labels (y):")
        print(f"  Shape: {data.y.shape}")
        print(f"  Device: {data.y.device}")

def debug_model_forward(self, data, smiles_batch=None):
    print("\nModel Forward Pass Debug:")
    print("-" * 50)
    print(f"Processing batch at {time.strftime('%H:%M:%S')}")
    print("\nInput Data:")
    print(f"Graph data shape: {data.x.shape if hasattr(data, 'x') else 'No graph data'}")
    print("\nProcessing graph features...")
    graph_features = self.graph_processor(data)
    print(f"Graph features shape: {graph_features.shape}")
    if smiles_batch is not None:
        print("\nProcessing SMILES data...")
        try:
            smiles_features = self.smiles_processor(smiles_batch)
            print(f"SMILES features shape: {smiles_features.shape}")
            print("Applying cross-attention...")
            combined_features = self.cross_attention(graph_features, smiles_features)
            print(f"Combined features shape: {combined_features.shape}")
            print("Using combined classifier...")
            output = self.combined_classifier(torch.cat([graph_features, combined_features], dim=-1))
        except Exception as e:
            print(f"Error processing SMILES data: {str(e)}")
            print("Falling back to graph-only processing...")
            output = self.graph_classifier(graph_features)
    else:
        print("Using graph-only classifier...")
        output = self.graph_classifier(graph_features)
    
    print(f"Output shape: {output.shape}")
    return output
def check_smiles_usage(model, loader=None):
    print("\nChecking SMILES Usage:")
    print("-" * 50)
    print("\n1. Checking Model Configuration:")
    has_smiles_processor = hasattr(model, 'smiles_processor')
    has_cross_attention = hasattr(model, 'cross_attention')
    print(f"Model has SMILES processor: {'Yes' if has_smiles_processor else 'No'}")
    print(f"Model has cross attention: {'Yes' if has_cross_attention else 'No'}")
    has_smiles = False
    if loader is not None:
        print("\n2. Checking Dataloader:")
        try:
            first_batch = next(iter(loader))
            has_smiles = isinstance(first_batch, tuple) and len(first_batch) == 2
            print(f"Dataloader provides SMILES data: {'Yes' if has_smiles else 'No'}")
        except Exception as e:
            print(f"Error checking dataloader: {str(e)}")
            print("Assuming no SMILES data in dataloader")
    else:
        print("\n2. Dataloader Check: Skipped (no loader provided)")
    print("\n3. Model Architecture:")
    for name, module in model.named_children():
        print(f"  - {name}: {type(module).__name__}")
    return {
        'model_has_processor': has_smiles_processor,
        'model_has_attention': has_cross_attention,
        'dataloader_has_smiles': has_smiles if loader is not None else None
    }

def train_and_evaluate(model, loaders, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    best_model = None
    best_metrics = None
    early_stopping = EarlyStopping(patience=10)
    device_manager = DeviceManager(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        for batch_idx, batch in enumerate(tqdm(loaders['train'], desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            try:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch) 
                if output is None:
                    print(f"Warning: Null output for batch {batch_idx}")
                    continue
                output = output.view(-1)
                targets = batch.y.float().view(-1)
                loss = criterion(output, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                train_loss += loss.item()
                train_preds.extend(output.detach().cpu().numpy())
                train_labels.extend(targets.cpu().numpy())
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        avg_train_loss = train_loss / len(loaders['train'])
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loaders['val'], desc="Validation")):
                try:
                    batch = batch.to(device)
                    output = model(batch)
                    if output is None:
                        continue
                        
                    output = output.view(-1)
                    targets = batch.y.float().view(-1)
                    loss = criterion(output, targets)
                    val_loss += loss.item()
                    val_preds.extend(output.cpu().numpy())
                    val_labels.extend(targets.cpu().numpy())
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        avg_val_loss = val_loss / len(loaders['val'])
        if scheduler is not None:
            scheduler.step()
        if len(train_preds) > 0 and len(val_preds) > 0:
            train_metrics = compute_metrics(np.array(train_labels), np.array(train_preds))
            val_metrics = compute_metrics(np.array(val_labels), np.array(val_preds))
            
           
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print("Training:", format_metrics(train_metrics, "Train"))
            
            print("Validation:", format_metrics(val_metrics, "Valid"))
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = {k: v.cpu() for k, v in model.state_dict().items()}
                best_metrics = val_metrics
                print(f"\nNew best model saved! (val_loss: {best_val_loss:.4f})")
        
      
        if early_stopping(avg_val_loss):
            print("Early stopping triggered")
            break
    
    return best_model, best_val_loss, best_metrics

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            try:
                data = data.to(DEVICE)
                output = model(data)
                if isinstance(output, tuple):
                    output = output[0]  
                
                if output is None:
                    print(f"Warning: Null output for batch {batch_idx}")
                    continue 
                loss = criterion(output.view(-1), data.y.float())
                test_loss += loss.item()
                preds = torch.sigmoid(output).cpu().numpy().flatten()
                labels = data.y.cpu().numpy().flatten()
                test_preds.extend(preds)
                test_labels.extend(labels)
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    if not test_preds:
        raise ValueError("No predictions were generated during testing")
    test_loss /= len(test_loader)
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_metrics = compute_metrics(test_labels, test_preds)
    return test_loss, test_metrics, test_preds, test_labels

def evaluate_multiple_test_sets(model, test_loaders, criterion):
    results = {}
    for name, loader in test_loaders.items():
        test_loss, test_metrics, test_pred, test_label = test_model(model, loader, criterion)
        results[name] = {
            'loss': test_loss,
            'metrics': test_metrics,
            'predictions': test_pred,
            'labels': test_label
        }
        print(f"\nResults for {name}:")
        print(f'Loss: {test_loss:.6f}')
        print(format_metrics(test_metrics, name))
    return results


def objective(trial, num_node_features, num_edge_features, train_loader,num_edge_types,num_node_types, val_loader, num_epochs):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)

    model =DSSA(
        in_dim=num_node_features,
        hidden_dim=64,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        num_classes=1,
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        processing_steps=4
    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    _, best_val_loss, _, _, _, _, _ = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10
    )
    return best_val_loss  


###########################################################################################################################

def analyze_target_distribution(dataset):
    y_values = [data.y.item() for data in dataset]
    mean = np.mean(y_values)
    median = np.median(y_values)
    std = np.std(y_values)
    skewness = stats.skew(y_values)
    kurtosis = stats.kurtosis(y_values)
    print(f"Target Distribution Statistics:")
    print(f"Mean: {mean:.4f}")
    print(f"Median: {median:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    plt.figure(figsize=(10, 6))
    plt.hist(y_values, bins=50, edgecolor='black')
    plt.title("Distribution of Target Values")
    plt.xlabel("Target Value")
    plt.ylabel("Frequency")
    plt.savefig("target_distribution.png")
    plt.close()
    return y_values
##################################################################################################

def extract_dataset(dataloader):
    if dataloader is None:
        return None
    return dataloader.dataset

def create_optimizer_and_scheduler(model, train_loader, num_epochs):
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.003,  
        total_steps=total_steps,  
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,  
        final_div_factor=1000.0,
        verbose=True
    )
    return optimizer, scheduler

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def setup_cuda_device():
    global DEVICE  
    print("\nChecking CUDA Setup...")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Checking possible issues:")
        print(f"\n1. PyTorch Installation:")
        print(f"   - PyTorch version: {torch.__version__}")
        try:
            from torch.backends import cudnn
            print(f"   - CUDA enabled in PyTorch build: {torch.backends.cudnn.enabled}")
        except ImportError:
            print("   - PyTorch may not be built with CUDA support")
        try:
            import subprocess
            nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
            print("\n2. NVIDIA Driver:")
            print(f"   - NVIDIA SMI available: Yes")
            print(f"   - Driver output: {nvidia_smi.decode().split('n')[0]}")
        except Exception as e:
            print("\n2. NVIDIA Driver:")
            print(f"   - Error running nvidia-smi: {str(e)}")
            print(f"   - Please check if NVIDIA drivers are installed")
            print("\n3. Environment Variables:")
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
        print(f"   - CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        DEVICE = torch.device('cpu')
        return DEVICE
    
    try:
        gpu_id = 0
        if torch.cuda.device_count() > 1:
            free_memory = []
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_memory.append(torch.cuda.memory_reserved())
            gpu_id = free_memory.index(min(free_memory))
        
        DEVICE = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(DEVICE)
        print("\nCUDA Setup Successful:")
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"GPU ID: {gpu_id}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Version: {torch.backends.cudnn.version()}")
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"\nMemory Status:")
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Reserved Memory: {reserved_memory:.2f} GB")
        print(f"Allocated Memory: {allocated_memory:.2f} GB")
        print(f"Free Memory: {(total_memory - reserved_memory):.2f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        return DEVICE
    except Exception as e:
        print(f"\nError during CUDA setup: {str(e)}")
        print("Falling back to CPU")
        DEVICE = torch.device('cpu')
        return DEVICE

def initialize_training_device():
    global DEVICE  
    DEVICE = setup_cuda_device()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)
        if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        print("\nCUDA Memory Management:")
        print("- Enabled gradient computation")
        print("- Cleared CUDA cache")
        print("- Set memory allocation configuration")
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("- Enabled TF32 for Ampere+ GPUs")
    else:
        print("\nUsing CPU for training (CUDA not available)")
        print("Note: Training will be significantly slower on CPU")
    
    return DEVICE

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return (bce_loss + focal_loss).mean()


def optimize_memory():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)  
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True



class PrefetchLoader:
    """Prefetch data to GPU for faster training"""
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device   
    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        
        for batch in self.loader:
            with torch.cuda.stream(stream):
                batch = batch.to(self.device, non_blocking=True)
            
            if not first:
                yield batch
            else:
                first = False
            
            stream.synchronize()
            
    def __len__(self):
        return len(self.loader)


def create_optimized_model_config():
    return {
        'batch_size': 256,  
        'learning_rate': 3e-4,  
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'num_workers': 4,
        'pin_memory': True
    }

#############################################################

def display_metrics_summary(results, all_predictions, all_labels):
    metrics_data = []
    for dataset_name, result in results.items():
        metrics = result['metrics']
        loss = result['loss']
        
        metrics_data.append({
            'Dataset': dataset_name,
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-score': f"{metrics['f1']:.3f}",
            'ROC-AUC': f"{metrics['roc_auc']:.3f}",
            'PR-AUC': f"{metrics['pr_auc']:.3f}",  
            'Sensitivity': f"{metrics['sensitivity']:.3f}",
            'Specificity': f"{metrics['specificity']:.3f}",
            'R2': f"{metrics['r2']:.3f}",
            'Loss': f"{loss:.3f}"
        })
    
    df = pd.DataFrame(metrics_data)
    print("Confusion Matrices:")
    print("=" * 50)
    for dataset_name in results.keys():
        if dataset_name in all_predictions and dataset_name in all_labels:
            pred_labels = (np.array(all_predictions[dataset_name]) > 0.5).astype(int)
            true_labels = np.array(all_labels[dataset_name])
            cm = confusion_matrix(true_labels, pred_labels)
            print(f"\n{dataset_name} Confusion Matrix:")
            print(cm)
            print("-" * 50)
    print("\nMetrics Summary:")
    print("=" * 150)  
    header = "Dataset    "
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "R2", "Loss"]
    for metric in metrics:
        header += f"{metric:<12}"
    print(header)
    print("-" * 150)  
    for _, row in df.iterrows():
        line = f"{row['Dataset']:<10}"
        for metric in metrics:
            line += f"{row[metric]:<12}"
        print(line)
    print("=" * 150) 
    return df


def save_results(results, all_predictions, all_labels, result_path):
    try:
        metrics_df = display_metrics_summary(results, all_predictions, all_labels)
        metrics_df.to_csv(os.path.join(result_path, 'metrics_summary.csv'), index=False)
        
        with open(os.path.join(result_path, 'metrics_summary.txt'), 'w') as f:
            f.write("Confusion Matrices:\n")
            f.write("=" * 50 + "\n")
            for dataset_name in results.keys():
                if dataset_name in all_predictions and dataset_name in all_labels:
                    pred_labels = (np.array(all_predictions[dataset_name]) > 0.5).astype(int)
                    true_labels = np.array(all_labels[dataset_name])
                    cm = confusion_matrix(true_labels, pred_labels)
                    f.write(f"\n{dataset_name} Confusion Matrix:\n")
                    f.write(str(cm) + "\n")
                    f.write("-" * 50 + "\n")
    
            f.write("\nMetrics Summary:\n")
            f.write("=" * 130 + "\n")
            metrics_df.to_string(f, index=False)
            f.write("\n" + "=" * 130 + "\n")
        plt.figure(figsize=(20, 20))
        plt.subplot(2, 2, 1)
        for i, (name, preds) in enumerate(all_predictions.items()):
            if name in all_labels:
                plt.subplot(2, len(all_predictions), i + 1)
                pred_labels = (np.array(preds) > 0.5).astype(int)
                cm = confusion_matrix(all_labels[name], pred_labels)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
        
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(result_path, 'confusion_matrices.png'), 
                    bbox_inches='tight', 
                    pad_inches=0.5)
        plt.close()
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        for name, preds in all_predictions.items():
            if name in all_labels:
                fpr, tpr, _ = roc_curve(all_labels[name], preds)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.subplot(1, 2, 2)
        for name, preds in all_predictions.items():
            if name in all_labels:
                precision, recall, _ = precision_recall_curve(all_labels[name], preds)
                pr_auc = average_precision_score(all_labels[name], preds)
                plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left')
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(result_path, 'performance_curves.png'),
                    bbox_inches='tight',
                    pad_inches=0.5)
        plt.close()
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'PR-AUC']
        plot_data = metrics_df[['Dataset'] + metrics_to_plot].copy()
        plot_data[metrics_to_plot] = plot_data[metrics_to_plot].astype(float)
        plt.figure(figsize=(15, 8))
        x = np.arange(len(plot_data['Dataset']))
        width = 0.15
        for i, metric in enumerate(metrics_to_plot):
            plt.bar(x + i*width, plot_data[metric], width, label=metric)


        plt.xlabel('Datasets')
        plt.ylabel('Score')
        plt.title('Comparison of Metrics Across Datasets')
        plt.xticks(x + width*2.5, plot_data['Dataset'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, 'metrics_comparison.png'),
                    bbox_inches='tight',
                    pad_inches=0.5)
        plt.close()
        
    except Exception as e:
        logging.error(f"Error in save_results: {str(e)}")
        traceback.print_exc()

def setup_training(args, num_node_features, num_edge_features, num_node_types, num_edge_types, device, train_loader):
    model = DSSA(
    in_dim=num_node_features,
    hidden_dim=128, 
    num_layers=4,
    num_heads=4,
    dropout=0,   
    num_classes=1,
    num_node_types=num_node_types,
    num_edge_types=num_edge_types,
    processing_steps=3  
).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    criterion = CombinedLoss(alpha=0.25, gamma=2.0)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.num_epochs
    scheduler = OneCycleLR(
    optimizer,
    max_lr=5e-4,   
    total_steps=total_steps,
    pct_start=0.3,  
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0  
)
    return model, optimizer, criterion, scheduler

###############################################################


def main():
    try:
      
        args = parse_args()
        set_seed(42)
        global DEVICE
        DEVICE = initialize_training_device()
        print(f"Training will use device: {DEVICE}")
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(args.result_path, f'run_{run_timestamp}')
        os.makedirs(result_path, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(result_path, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)
        logging.info("Loading datasets...")
        data_result = dataload()
        if len(data_result) != 12: 
            raise ValueError(f"Expected 12 values from dataload, got {len(data_result)}")

        (train_dataset, val_dataset, test_dataset, 
        ts1_dataset, ts2_dataset, ts3_dataset,
        num_node_features, num_edge_features,
        num_node_types, num_edge_types, class_weights,
        smiles_config) = data_result  
        loaders = create_optimized_dataloaders([train_dataset, val_dataset, test_dataset,
                                            ts1_dataset, ts2_dataset, ts3_dataset])

        logging.info(f"Data loaded successfully:")
        logging.info(f"Number of node features: {num_node_features}")
        logging.info(f"Number of edge features: {num_edge_features}")
        logging.info(f"Number of node types: {num_node_types}")
        logging.info(f"Number of edge types: {num_edge_types}")

      
        logging.info("Creating dataloaders...")
        datasets = [train_dataset, val_dataset, test_dataset, 
                   ts1_dataset, ts2_dataset, ts3_dataset]
        try:
            loaders = create_optimized_dataloaders(
                datasets=datasets,
                batch_size=256,
                num_workers=4
            )
            logging.info("Dataloaders created successfully")
        except Exception as e:
            logging.error(f"Error creating dataloaders: {str(e)}")
            raise
        logging.info("Setting up model and training components...")
        model, optimizer, criterion, scheduler = setup_training(
            args=args,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            device=DEVICE,
            train_loader=loaders['train']
        )  
        logging.info("Checking SMILES integration...")
        smiles_status = check_smiles_usage(model, loaders.get('train'))
        if smiles_status['dataloader_has_smiles'] is not None:
            if not smiles_status['dataloader_has_smiles']:
                logging.warning("No SMILES data found in dataloader!")
                logging.info("Model will operate in graph-only mode")
            else:
                logging.info("SMILES data found and will be used in training")
        else:
            logging.warning("Could not check dataloader for SMILES data")
        
        logging.info("Starting training...")
        best_model, best_val_loss, best_metrics = train_and_evaluate(
            model=model,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            device=DEVICE
        )
        model_save_path = os.path.join(result_path, 'best_model.pth')
        torch.save({
            'model_state_dict': best_model,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'model_config': {
                'in_dim': num_node_features,
                'hidden_dim': 128,
                'num_layers': 4,
                'num_heads': 4,
                'dropout': 0,
                'num_classes': 1,
                'num_node_types': num_node_types,
                'num_edge_types': num_edge_types,
                'processing_steps': 4
            }
        }, model_save_path)
        logging.info("Running final evaluation...")
        model.load_state_dict(best_model)
        test_loaders = {
            'test': loaders['test'],
            'ts1': loaders['ts1'],
            'ts2': loaders['ts2'],
            'ts3': loaders['ts3']
        }
        
        results = {}
        all_predictions = {}
        all_labels = {}

        for name, loader in test_loaders.items():
            if loader is None:
                continue

            logging.info(f"\nEvaluating on {name} dataset...")
            try:
                test_loss, metrics, predictions, labels = test_model(
                    model=model,
                    test_loader=loader,
                    criterion=criterion
                )

                results[name] = {
                    'loss': test_loss,
                    'metrics': metrics
                }
                all_predictions[name] = predictions
                all_labels[name] = labels

                logging.info(f"Results for {name}:")
                logging.info(f"Loss: {test_loss:.4f}")
                logging.info(format_metrics(metrics, name))

            except Exception as e:
                logging.error(f"Error evaluating {name}: {str(e)}")
                continue

        try:
            save_results(results, all_predictions, all_labels, result_path)
            logging.info(f"\nTraining complete. Results saved in {result_path}")
        except Exception as e:
            logging.error(f"Error saving final results: {str(e)}")
       
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        traceback.print_exc()

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Training session ended.")
        plt.close('all')

if __name__ == "__main__":
    main()
