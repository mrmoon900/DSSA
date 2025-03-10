import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data
import time

import os
from transformers import RobertaModel,AutoTokenizer
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

class SMILESProcessor(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.vocab = {
          
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8,
             '=': 9, '#': 10, ':': 11, '-': 12,
           
            '1': 13, '2': 14, '3': 15, '4': 16, '5': 17, '6': 18, '7': 19, '8': 20,
         
            '(': 21, ')': 22, '[': 23, ']': 24,
         
            '<PAD>': 25, '<UNK>': 26, '<START>': 27, '<END>': 28,
           
            '+': 29, '-': 30, '.': 31, '/': 32, '\\': 33,
          
            'H': 34, '@': 35, '@@': 36
        }
        
        self.vocab_size = len(self.vocab)
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=self.vocab['<PAD>'])
        self.pos_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def tokenize_smiles(self, smiles_string):
        tokens = []
        i = 0
        while i < len(smiles_string):
            if i + 1 < len(smiles_string) and smiles_string[i:i+2] in self.vocab:
                tokens.append(self.vocab[smiles_string[i:i+2]])
                i += 2
            else:
                token = smiles_string[i]
                tokens.append(self.vocab.get(token, self.vocab['<UNK>']))
                i += 1
        return torch.tensor(tokens)
    
    def forward(self, smiles_batch):
        token_lists = [self.tokenize_smiles(s) for s in smiles_batch]
        max_len = max(len(t) for t in token_lists)
        padded_tokens = torch.stack([
            F.pad(t, (0, max_len - len(t)), value=self.vocab['<PAD>'])
            for t in token_lists
        ])
        attention_mask = (padded_tokens != self.vocab['<PAD>']).float()
        x = self.embedding(padded_tokens)
        positions = torch.arange(max_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_embeddings = self.pos_encoder(x)
        x = x + pos_embeddings
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x,
            attention_mask.sum(1).cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        output, _ = self.gru(x_packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        masked_output = output * attention_mask.unsqueeze(-1)
        summed = masked_output.sum(dim=1)
        counts = attention_mask.sum(dim=1, keepdim=True)
        pooled = summed / counts
        final_output = self.output_proj(pooled)
        return final_output

class DSSA(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout, num_classes, num_node_types, num_edge_types, processing_steps):
        super().__init__()
        self.debug_mode = True
        self.graph_processor = GraphNodeRefine(
            in_dim, hidden_dim, num_layers, num_heads, 
            dropout, num_classes, num_node_types, 
            num_edge_types, processing_steps
        )
        self.smiles_processor = SMILESProcessor(hidden_dim, dropout)
        self.cross_attention = CrossModalTransformer(hidden_dim, num_heads, dropout)
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  
        )
        self.combined_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        graph_features = self.graph_processor(data)
        out = self.graph_classifier(graph_features)
        return out.squeeze(-1)  
    
class GraphNodeRefinement(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout, num_classes, num_node_types, num_edge_types, processing_steps):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.node_embedding = nn.Linear(in_dim, hidden_dim)        
        self.vocab_size = 38  
        self.text_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.text_encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.gat_layers = nn.ModuleList([
            GATConv(
                hidden_dim, 
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.node_embedding(x)
        if hasattr(data, 'node_types'):
            node_type_embed = self.node_type_embedding(data.node_types)
            h = h + node_type_embed
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            h_prev = h
            h = gat(h, edge_index)
            h = norm(h)
            h = h + h_prev
            h = F.relu(h)
        
        attention_weights = self.global_attention(h)
        attention_weights = F.softmax(attention_weights, dim=0)
        h_graph = torch.sum(h * attention_weights, dim=0)
        h_graph = global_mean_pool(h, batch)
        out = self.classifier(h_graph)
        return out

class NodeFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GATConv(
                hidden_dim, 
                hidden_dim // 8, 
                heads=8, 
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        return global_mean_pool(x, batch)

class StructuralFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        return global_mean_pool(x, batch)

class CrossModalTransformer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x1, x2):
        attended, _ = self.attention(x1.unsqueeze(0), x2.unsqueeze(0), x2.unsqueeze(0))
        attended = attended.squeeze(0)

        x = x1 + attended
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x
    
class GraphNodeRefine(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout, num_classes, num_node_types, num_edge_types, processing_steps):
        super().__init__()
        self.node_embedding = nn.Linear(in_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATConv(
                hidden_dim, 
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.node_embedding(x)
        h = self.dropout(h)
        if hasattr(data, 'node_types'):
            node_type_embed = self.node_type_embedding(data.node_types)
            h = h + node_type_embed
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            h_prev = h
            h = gat(h, edge_index)
            h = norm(h)
            h = h + h_prev  
            h = F.relu(h)
            h = self.dropout(h)
        attention_weights = self.global_attention(h)
        attention_weights = F.softmax(attention_weights, dim=0)
        h_graph = torch.sum(h * attention_weights, dim=0)
        h_graph = h_graph + global_mean_pool(h, batch)
        
        return h_graph
    
class ContrastiveLearning(nn.Module):
    def __init__(self, hidden_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x1, x2):
        z1 = F.normalize(self.projection(x1), dim=-1)
        z2 = F.normalize(self.projection(x2), dim=-1)
        similarity = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(similarity.size(0), device=similarity.device)
        loss = F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)
        
        return loss / 2
