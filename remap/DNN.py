import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import torch.optim as optim
from tqdm import trange


class DNN(nn.Module):
    """
    Deep Neural Network for training the location predictor and neighborhood gene-gene covariance predictor.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int, optional (default=2)
        Number of output features.
    hidden_dims : list of int, optional (default=[200, 100, 50])
        Sizes of hidden layers. If None or empty, defaults to [200, 100, 50].
    dropout : float, optional (default=0.25)
        Dropout probability.
    type_pred : str, optional (default="loc")
        Type of prediction:
        - "loc": output layer uses Sigmoid activation.
        - "covet": output layer is linear (no activation).
    **kwargs :
        Extra keyword arguments (currently unused).
    """
    
    def __init__(self,
                 in_channels,
                 out_channels = 2,
                 hidden_dims = None,
                 dropout = 0.25,
                 type_pred = "loc",  # "loc" → sigmoid at output
                 **kwargs) -> None:
        super(DNN, self).__init__()

        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = [200, 100, 50]

        self.type_pred = type_pred

        # Build hidden layers dynamically
        layers = []
        prev_dim = in_channels
        for i in range(len(hidden_dims)):
            dim = hidden_dims[i]
            layers.append(nn.Linear(prev_dim, dim))
            if dropout > 0 and i == 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            prev_dim = dim
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        if type_pred == "loc":
            self.output_layer = nn.Sequential(
                nn.Linear(prev_dim, out_channels),
                nn.Sigmoid()
            )
        else:
            self.output_layer = nn.Linear(prev_dim, out_channels)

    def forward(self, input):
        z = self.hidden_layers(input[0])
        out = self.output_layer(z)
        return [out, input, z]

    def loss_function(self, *args) -> dict:
        pred = args[0]
        input = args[1]
        target = input[1]
        mse_loss = F.mse_loss(pred, target)
        return {'loss': mse_loss}
    
    
class DNN_rel(nn.Module):
    """
    Deep Neural Network for training the relative distance predictor to predict relative distance between two cells.

    Parameters
    ----------
    in_channels : int
        Number of input features for each vector.
    hidden_dims : list of int, optional (default=[200, 100, 50])
        Sizes of hidden layers. Must contain at least two layers to support the residual structure.
    out_channels : int, optional (default=1)
        Number of outputs.
    dropout : float, optional (default=0.05)
        Dropout probability applied across layers to prevent overfitting.
    """
    
    def __init__(self, 
                 in_channels,
                 hidden_dims= None,
                 out_channels= 1,
                 dropout= 0.05):
        super(DNN_rel, self).__init__()

        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = [200, 100, 50]
        assert len(hidden_dims) >= 2, "At least two hidden layers required to maintain structure."

        self.shared_layer = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(p=dropout)

        self.fclayer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        # fclayer3: from hidden_dims[1] -> hidden_dims[1] (residual block)
        self.fclayer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[1], out_channels),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor):
        # Encode each feature vector
        z1 = self.shared_layer(input[0])
        z2 = self.shared_layer(input[1])

        # Compute absolute difference and apply dropout
        z_diff = self.dropout(torch.abs(z1 - z2))

        # Pass through layers with residual connection
        zL2 = self.dropout(self.fclayer2(z_diff))
        zL3 = zL2 + self.dropout(self.fclayer3(zL2))

        # Output layer with sigmoid * sqrt(2)
        result = self.output_layer(zL3) * math.sqrt(2)

        return [result, input]

    def loss_function(self, *args):
        dist_pred = args[0]
        dist_true = args[1][2]
        weight = args[1][3]
        se = ((dist_pred - dist_true) ** 2) * weight.unsqueeze(1)
        loss = torch.mean(se)
        return {'loss': loss}
    
    
class TrainerExe(object):
    """
    TrainerExe: A simple training utility for PyTorch models.

    This class manages the training loop for a given model and dataloader.
    It automatically handles device placement (CPU/GPU), optimizer setup,
    learning rate adjustments, and early stopping based on loss stagnation.

    Attributes
    ----------
    filename : str
        Filename for saving results.
    device : torch.device
        Automatically set to "cuda" if a GPU is available, otherwise "cpu".
    """
 
    def __init__(self, filename):
        super(TrainerExe, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.filename = filename
        
    def train(self, model, train_loader, break_rate = 5e-6, num_epochs=500, learning_rate=1e-3, RCcountMax = 40):
        self.learning_rate = min(1e-2, learning_rate)
        self.model = model.to(self.device)
        self.break_rate = break_rate
        
        optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate, 
                                    weight_decay=1e-5) 
        # if next(self.model.parameters()).is_cuda:
        #     print("Model is on GPU")
        # else:
        #     print("Model is on CPU")
        RCcount = 0
        loss_min = 99999999
  
        # pbar = tqdm(range(num_epochs), desc=f"Training, {self.filename}", position=0)
        pbar = trange(num_epochs, desc=f"Training, {self.filename}", position=0)
        stopped_early = False
        for epoch in range(num_epochs):
            total_loss = 0
            # for i, img in enumerate(tqdm(train_loader)):
            for i, img in enumerate(train_loader):
                img = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in img)
                recon = self.model(img)
                loss = self.model.loss_function(*recon)
                loss.get("loss").backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.get("loss").data
            pbar.set_postfix({"loss": f"{total_loss:.4f}"})
            pbar.update(1)

            if (total_loss>loss_min):
                RCcount = RCcount + 1
                if (RCcount == RCcountMax):
                    RCcount = 0
                    self.learning_rate = self.learning_rate/2
                    optimizer.param_groups[0]['lr'] = self.learning_rate
                    loss_min = loss_min + 10
            else:
                loss_min = total_loss
            if (self.learning_rate < self.break_rate):
                stopped_early = True
                # pbar.n = pbar.total
                # pbar.last_print_n = pbar.total
                remaining = num_epochs - (epoch + 1)
                if remaining > 0:
                    pbar.update(remaining)
                break
        pbar.close()  
        if stopped_early:
            print("Early stopping criteria reached.")
        torch.cuda.empty_cache()
