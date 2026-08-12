from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class ChargerDatasetLazy(Dataset):
    """Dataset for charger event sequences with lazy loading from HDF5."""
    
    def __init__(self, hdf5_path, charger_ids):
        """
        Args:
            hdf5_path: Path to HDF5 file
            charger_ids: List of charger IDs to include in this dataset
        """
        self.hdf5_path = hdf5_path
        self.charger_ids = charger_ids
        self._file = None
        
        # Build index: list of (charger_id, day_index) tuples
        # Each entry represents one sample (one day from one charger)
        # Batches will naturally mix samples from different chargers
        self.index = []
        with h5py.File(hdf5_path, 'r') as f:
            for charger_id in charger_ids:
                if charger_id in f:
                    num_days = f[charger_id]['labels'].shape[0]
                    # Filter out empty sequences
                    for day_idx in range(num_days):
                        cont_feat = f[charger_id]['cont_feat'][day_idx]
                        if len(cont_feat) > 0:  # Only include non-empty sequences
                            self.index.append((charger_id, day_idx))
        
        print(f"  Lazy dataset with {len(self.charger_ids)} chargers, {len(self.index)} valid samples")
    
    def _open_hdf5(self):
        """Open HDF5 file (one per worker process)."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """Load a single sample (one day) on-demand from HDF5."""
        self._open_hdf5()
        
        charger_id, day_idx = self.index[idx]
        
        # Load from HDF5 - only this specific day
        cont_feat_raw = self._file[charger_id]['cont_feat'][day_idx]
        label = self._file[charger_id]['labels'][day_idx]
        
        # Reshape to (seq_len, 3)
        cont_feat = cont_feat_raw.reshape(-1, 3)
        
        return {
            'cont_feat': torch.FloatTensor(cont_feat),
            'label': torch.FloatTensor([label])
        }
    
    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if self._file is not None:
            self._file.close()

    def get_all_labels(self):
        """Get all labels from the dataset for class weight calculation."""
        all_labels = []
        with h5py.File(self.hdf5_path, 'r') as f:
            for charger_id in self.charger_ids:
                if charger_id in f:
                    all_labels.extend(f[charger_id]['labels'][:].tolist())
        return all_labels


# Keep the old in-memory dataset for comparison
class ChargerDataset(Dataset):
    """Dataset for charger event sequences (loads all data into memory)."""
    
    def __init__(self, charger_data):
        """
        Args:
            charger_data: Dict with charger_id as keys, each containing 'cont_feat' and 'labels'
        """
        self.samples = []
        
        for charger_id, data in charger_data.items():
            cont_feats = data['cont_feat']
            labels = data['labels']
            
            for feat, label in zip(cont_feats, labels):
                if len(feat) > 0:  # Skip empty sequences
                    self.samples.append({
                        'cont_feat': torch.FloatTensor(feat),  # Shape: (seq_len, 3)
                        'label': torch.FloatTensor([label])
                    })
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_all_labels(self):
        """Get all labels from the dataset for class weight calculation."""
        return [sample['label'].item() for sample in self.samples]

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

class ChargerLSTM(nn.Module):
    """LSTM model for predicting charger failure from event sequences."""
    
    def __init__(self, config):
        """
        Args:
            config: Dict with keys:
                - vocab_size: Number of unique event types (for embedding)
                - embed_dim: Embedding dimension for event IDs
                - hidden_dims: List of hidden dimensions for each LSTM layer
                - dropout: Dropout probability
        """
        super(ChargerLSTM, self).__init__()
        
        self.vocab_size = config['vocab_size']
        self.embed_dim = config['embed_dim']
        self.hidden_dims = config['hidden_dims']
        self.dropout = config['dropout']
        
        # Embedding layer for event IDs (first feature)
        self.event_embedding = nn.Embedding(
            num_embeddings=self.vocab_size + 1,  # +1 for padding (0)
            embedding_dim=self.embed_dim,
            padding_idx=0
        )
        
        # LSTM layers with explicit dropout between them
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        input_size = self.embed_dim + 2  # Embedding + sin + cos

        for i, hidden_dim in enumerate(self.hidden_dims):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_dim,
                    batch_first=True,
                    # Don't use LSTM's internal dropout - we add explicit layers instead
                )
            )
            # Add dropout after each LSTM except the last
            if i < len(self.hidden_dims) - 1:
                self.dropout_layers.append(nn.Dropout(self.dropout))

            input_size = hidden_dim

        # Classification head
        self.fc = nn.Linear(self.hidden_dims[-1], 1)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, 3) where features are [event_id, sin, cos]
            mask: (batch_size, seq_len) boolean mask for valid positions
        
        Returns:
            logits: (batch_size,) prediction logits
        """
        batch_size, seq_len, _ = x.shape
        
        # Split features
        event_ids = x[:, :, 0].long()  # (batch_size, seq_len)
        time_feats = x[:, :, 1:]       # (batch_size, seq_len, 2)
        
        # Embed event IDs
        event_embeds = self.event_embedding(event_ids)  # (batch_size, seq_len, embed_dim)
        
        # Concatenate embedded events with time features
        lstm_input = torch.cat([event_embeds, time_feats], dim=-1)  # (batch_size, seq_len, embed_dim+2)
        
        # Pass through LSTM layers with dropout between them
        for i, lstm in enumerate(self.lstm_layers):
            lstm_input, _ = lstm(lstm_input)
            # Apply dropout after LSTM (except for the last one)
            if i < len(self.lstm_layers) - 1:
                lstm_input = self.dropout_layers[i](lstm_input)
        
        # Use the last valid output for each sequence
        if mask is not None:
            # Get the index of the last valid position for each sequence
            lengths = mask.sum(dim=1) - 1  # (batch_size,)
            lengths = lengths.clamp(min=0)
            
            # Gather the last valid hidden state
            idx = lengths.unsqueeze(1).unsqueeze(2).expand(-1, -1, lstm_input.size(2))
            last_output = lstm_input.gather(1, idx).squeeze(1)  # (batch_size, hidden_dim)
        else:
            last_output = lstm_input[:, -1, :]  # (batch_size, hidden_dim)
        
        # Classification
        out = self.dropout_layer(last_output)
        logits = self.fc(out).squeeze(-1)  # (batch_size,)
        
        return logits

def collate_fn(batch):
    """Collate function to pad variable-length sequences."""
    cont_feats = pad_sequence([b['cont_feat'] for b in batch], batch_first=True, padding_value=0.0)
    labels = torch.stack([b['label'] for b in batch]).squeeze(-1)
    
    # Create mask: True for valid positions, False for padding
    lengths = torch.tensor([len(b['cont_feat']) for b in batch])
    max_len = cont_feats.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    
    return cont_feats, labels, mask

def split_chargers_for_lazy_loading(hdf5_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split chargers into train/val/test sets for lazy loading.
    
    Args:
        hdf5_path: Path to HDF5 file
        train_ratio: Ratio of chargers for training
        val_ratio: Ratio of chargers for validation
        test_ratio: Ratio of chargers for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        train_chargers, val_chargers, test_chargers: Lists of charger IDs
        charger_stats: DataFrame with statistics per charger
    """
    # Analyze charger statistics
    charger_stats = []
    with h5py.File(hdf5_path, "r") as f:
        all_chargers = list(f.keys())
        
        for charger in all_chargers:
            labels = f[charger]["labels"][:]
            cont_feat_raw = f[charger]["cont_feat"][:]
            
            # Filter valid days
            valid_days = [i for i, feat in enumerate(cont_feat_raw) if len(feat) > 0]
            num_days = len(valid_days)
            
            if num_days > 0:
                valid_labels = labels[valid_days]
                num_events = sum(len(cont_feat_raw[i]) for i in valid_days) // 3  # Divide by 3 since flattened
                num_positive = sum(valid_labels)
                
                charger_stats.append({
                    'charger': charger,
                    'num_days': num_days,
                    'num_events': num_events,
                    'num_positive': num_positive,
                    'pos_ratio': num_positive / num_days if num_days > 0 else 0
                })
    
    charger_stats = pd.DataFrame(charger_stats).sort_values('num_events', ascending=False)
    
    print("Charger Statistics:")
    print(f"Total chargers: {len(charger_stats)}")
    print(f"Total days: {charger_stats['num_days'].sum()}")
    print(f"Total events: {charger_stats['num_events'].sum()}")
    print(f"Positive labels: {charger_stats['num_positive'].sum()}")
    print(f"\nEvents per charger - Min: {charger_stats['num_events'].min()}, "
          f"Median: {charger_stats['num_events'].median():.0f}, "
          f"Max: {charger_stats['num_events'].max()}")
    
    # Split chargers stratified by activity level
    chargers = charger_stats['charger'].tolist()
    
    # Use stratified split based on event count quantiles
    charger_stats['activity_bin'] = pd.qcut(charger_stats['num_events'], q=3, 
                                             labels=['low', 'medium', 'high'], duplicates='drop')
    
    train_chargers, temp_chargers = train_test_split(
        chargers, test_size=(val_ratio + test_ratio), 
        random_state=random_seed,
        stratify=charger_stats['activity_bin']
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_chargers, test_chargers = train_test_split(
        temp_chargers, test_size=(1 - val_size),
        random_state=random_seed
    )
    
    # Print split info
    train_stats = charger_stats[charger_stats['charger'].isin(train_chargers)]
    val_stats = charger_stats[charger_stats['charger'].isin(val_chargers)]
    test_stats = charger_stats[charger_stats['charger'].isin(test_chargers)]
    
    print(f"\nSplit:")
    print(f"Train: {len(train_chargers)} chargers, {train_stats['num_days'].sum()} days")
    print(f"Val: {len(val_chargers)} chargers, {val_stats['num_days'].sum()} days")
    print(f"Test: {len(test_chargers)} chargers, {test_stats['num_days'].sum()} days")
    
    return train_chargers, val_chargers, test_chargers, charger_stats

def split_data_by_chargers(hdf5_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data by chargers into train/val/test sets (loads all into memory).
    
    Args:
        hdf5_path: Path to HDF5 file
        train_ratio: Ratio of chargers for training
        val_ratio: Ratio of chargers for validation
        test_ratio: Ratio of chargers for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        train_data, val_data, test_data: Dicts with charger data
    """
    # Load all data
    with h5py.File(hdf5_path, "r") as f:
        all_chargers = list(f.keys())
        
        per_charger = {}
        for charger in all_chargers:
            cont_feat_raw = f[charger]["cont_feat"][:]
            labels = f[charger]["labels"][:]
            
            # Reshape cont_feat to (num_days, seq_len, 3)
            cont_feat = [el.reshape(-1, 3) for el in cont_feat_raw if len(el) > 0]
            
            # Filter out days with no events
            valid_indices = [i for i, feat in enumerate(cont_feat_raw) if len(feat) > 0]
            valid_labels = labels[valid_indices]
            
            if len(cont_feat) > 0:
                per_charger[charger] = {
                    'cont_feat': cont_feat,
                    'labels': valid_labels
                }
    
    # Analyze charger statistics
    charger_stats = []
    for charger, data in per_charger.items():
        num_days = len(data['labels'])
        num_events = sum(len(feat) for feat in data['cont_feat'])
        num_positive = sum(data['labels'])
        charger_stats.append({
            'charger': charger,
            'num_days': num_days,
            'num_events': num_events,
            'num_positive': num_positive,
            'pos_ratio': num_positive / num_days if num_days > 0 else 0
        })
    
    charger_stats = pd.DataFrame(charger_stats).sort_values('num_events', ascending=False)
    
    print("Charger Statistics:")
    print(f"Total chargers: {len(charger_stats)}")
    print(f"Total days: {charger_stats['num_days'].sum()}")
    print(f"Total events: {charger_stats['num_events'].sum()}")
    print(f"Positive labels: {charger_stats['num_positive'].sum()}")
    print(f"\nEvents per charger - Min: {charger_stats['num_events'].min()}, "
          f"Median: {charger_stats['num_events'].median():.0f}, "
          f"Max: {charger_stats['num_events'].max()}")
    
    # Split chargers stratified by activity level
    chargers = charger_stats['charger'].tolist()
    
    # Use stratified split based on event count quantiles
    charger_stats['activity_bin'] = pd.qcut(charger_stats['num_events'], q=3, 
                                             labels=['low', 'medium', 'high'], duplicates='drop')
    
    train_chargers, temp_chargers = train_test_split(
        chargers, test_size=(val_ratio + test_ratio), 
        random_state=random_seed,
        stratify=charger_stats['activity_bin']
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_chargers, test_chargers = train_test_split(
        temp_chargers, test_size=(1 - val_size),
        random_state=random_seed
    )
    
    # Create data splits
    train_data = {c: per_charger[c] for c in train_chargers}
    val_data = {c: per_charger[c] for c in val_chargers}
    test_data = {c: per_charger[c] for c in test_chargers}
    
    print(f"\nSplit:")
    print(f"Train: {len(train_chargers)} chargers, "
          f"{sum(len(d['labels']) for d in train_data.values())} days")
    print(f"Val: {len(val_chargers)} chargers, "
          f"{sum(len(d['labels']) for d in val_data.values())} days")
    print(f"Test: {len(test_chargers)} chargers, "
          f"{sum(len(d['labels']) for d in test_data.values())} days")
    
    return train_data, val_data, test_data, charger_stats
