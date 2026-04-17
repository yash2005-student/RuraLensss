"""
Graph Neural Network for Infrastructure Impact Prediction.

Reference architecture:
- Layer 1: GCNConv(24 -> 128)
- Layer 2: GATConv(128 -> 128 via 4 heads x 32, edge_dim=1) + residual from Layer 1
- Layer 3: GCNConv(128 -> 128) + residual from Layer 1
- Layer 4: Linear(128 -> 12)
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples"""
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class InfrastructureGNN(nn.Module):
    """
    GNN architecture for 12-dimensional multi-label node impact prediction.
    """

    def __init__(self, input_dim=24, hidden_dim=128, output_dim=12, dropout=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Layer 1: GCNConv (24 -> 128)
        self.conv1 = GCNConv(input_dim, hidden_dim)

        # Layer 2: GATConv (128 -> 32 x 4 heads = 128)
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 4,
            heads=4,
            concat=True,
            edge_dim=1,
            dropout=dropout,
        )

        # Layer 3: GCNConv (128 -> 128)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Layer 4: Linear output projection (128 -> 12 logits)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Forward pass through the GNN
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            batch: Batch assignment for graph batching
            
        Returns:
            Node-level predictions [num_nodes, output_dim]
        """
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1),
                dtype=x.dtype,
                device=x.device,
            )

        # GATConv with edge_dim=1 expects [num_edges, 1]
        if edge_weight.dim() == 1:
            edge_attr = edge_weight.unsqueeze(-1)
        else:
            edge_attr = edge_weight
            edge_weight = edge_weight.squeeze(-1)

        # Layer 1
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        # Layer 2 + residual from Layer 1
        x2 = self.conv2(x1, edge_index, edge_attr=edge_attr)
        x2 = x2 + x1
        x2 = F.relu(x2)
        x2 = self.dropout(x2)

        # Layer 3 + residual from Layer 1
        x3 = self.conv3(x2, edge_index, edge_weight=edge_weight)
        x3 = x3 + x1
        x3 = F.relu(x3)
        x3 = self.dropout(x3)

        # Layer 4 logits (no activation)
        logits = self.fc_out(x3)
        return logits


class ImpactPredictor:
    """
    Wrapper class for training and inference
    """
    
    def __init__(
        self,
        model_path=None,
        learning_rate=0.001,
        device=None,
        use_focal_loss=False,
        temperature=1.0,
        status_veto_weight=2.5,
        dropout=0.3,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Temperature scaling for sharper/softer predictions
        self.temperature = temperature

        # Kept for backward compatibility with existing script signatures.
        self.status_veto_weight = status_veto_weight

        self.model = InfrastructureGNN(dropout=dropout).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
            min_lr=1e-6,
        )

        # Loss function selection
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=torch.tensor([5.0]))
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        if model_path:
            self.load_model(model_path)

    def compute_weighted_loss(self, logits, targets):
        """Weighted BCE loss with 3x focus on critical nodes."""
        loss_per_node = self.criterion(logits, targets)
        weights = torch.ones_like(targets)
        critical_mask = targets.mean(dim=1) > 0.5
        weights[critical_mask] = 3.0
        return (loss_per_node * weights).mean()

    def train_step(self, data_batch):
        """Single training step with weighted loss"""
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        data_batch = data_batch.to(self.device)

        # Forward pass
        out = self.model(data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch)

        # Compute weighted loss
        weighted_loss = self.compute_weighted_loss(out, data_batch.y)

        # Backward pass
        weighted_loss.backward()

        # Prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return weighted_loss.item()

    def validation_step(self, data_batch):
        """Single validation step with weighted loss."""
        self.model.eval()
        with torch.no_grad():
            data_batch = data_batch.to(self.device)
            out = self.model(data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch)
            loss = self.compute_weighted_loss(out, data_batch.y)
        return loss.item()
    
    def predict(self, x, edge_index, edge_weight=None):
        """
        Predict impact for a single graph (returns probabilities).
        
        Args:
            x: Node features [num_nodes, 24]
            edge_index: Edge connections [2, num_edges]
            edge_weight: Edge weights [num_edges]
            
        Returns:
            Impact probabilities [num_nodes, 12] (values 0.0-1.0)
        """
        self.model.eval()
        
        # Convert to tensors if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        if edge_weight is not None and not isinstance(edge_weight, torch.Tensor):
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        with torch.no_grad():
            logits = self.model(x, edge_index, edge_weight)

            # Apply temperature scaling (T < 1.0 = sharper, T > 1.0 = softer)
            calibrated_logits = logits / self.temperature

            # Apply sigmoid to get probabilities (0-1 range)
            probabilities = torch.sigmoid(calibrated_logits)

        return probabilities.cpu().numpy()
    
    def predict_with_threshold(self, x, edge_index, edge_weight=None, threshold=0.5):
        """
        Predict impact with inference-time threshold (decision boundary).
        
        ⚠️ CRITICAL: Threshold is applied ONLY at inference, not training!
        The model outputs objective probabilities. The threshold determines
        how we INTERPRET those probabilities for alerts/decisions.
        
        Args:
            x: Node features [num_nodes, 24]
            edge_index: Edge connections [2, num_edges]
            edge_weight: Edge weights [num_edges]
            threshold: Decision boundary (default: 0.5)
                      - Lower (0.3): More sensitive, more alerts
                      - Higher (0.7): Less sensitive, fewer alerts
            
        Returns:
            probabilities: [num_nodes, 12] - Raw probability scores (0-1)
            alerts: [num_nodes, 12] - Boolean alerts (>= threshold)
            risk_level: str - Overall risk assessment
        """
        # Get raw probabilities
        probabilities = self.predict(x, edge_index, edge_weight)
        
        # Apply threshold (inference-time decision boundary)
        alerts = (probabilities >= threshold).astype(int)
        
        # Compute overall risk level based on impact probability (dimension 0)
        impact_probs = probabilities[:, 0]
        max_impact = impact_probs.max()
        
        if max_impact >= 0.7:
            risk_level = "🔴 CRITICAL"
        elif max_impact >= 0.5:
            risk_level = "🟠 HIGH"
        elif max_impact >= 0.3:
            risk_level = "🟡 MODERATE"
        else:
            risk_level = "🟢 LOW"
        
        return probabilities, alerts, risk_level
    
    def set_temperature(self, temperature):
        """
        Adjust temperature scaling for inference.
        
        Args:
            temperature: float
                - T = 1.0: Default (no scaling)
                - T = 0.5: Sharper predictions (crisis mode)
                - T = 0.3: Emergency mode (very sharp)
                - T = 2.0: Softer predictions (conservative)
        """
        self.temperature = temperature
        print(f"🌡️ Temperature set to {temperature:.2f}")
        if temperature < 0.5:
            print("⚠️ Emergency mode: Very sharp predictions")
        elif temperature < 1.0:
            print("🔥 Crisis mode: Sharper predictions")
        elif temperature > 1.0:
            print("❄️ Conservative mode: Softer predictions")

    def save_model(self, path, metadata=None):
        """Save model checkpoint"""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metadata": metadata or {},
        }

        torch.save(checkpoint, str(model_path))
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            print(f"⚠️ Strict load failed, falling back to non-strict load: {exc}")
            self.model.load_state_dict(state_dict, strict=False)

        optimizer_state = checkpoint.get("optimizer_state_dict")
        scheduler_state = checkpoint.get("scheduler_state_dict")

        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except ValueError as exc:
                print(f"⚠️ Skipping optimizer state due to mismatch: {exc}")

        if scheduler_state is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state)
            except ValueError as exc:
                print(f"⚠️ Skipping scheduler state due to mismatch: {exc}")

        print(f"Model loaded from {path}")
