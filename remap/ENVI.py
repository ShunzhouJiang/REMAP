import numpy as np
import sklearn.neighbors
import scanpy as sc
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from .ENVI_utils import *
import pickle
from scipy.sparse import issparse

class ENVI(nn.Module):

    """
    The code is rewritten from ENVI. ENVI integrates spatial and single-cell data

    Parameters:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        sc_data (anndata): anndata with single cell data
        spatial_key (str): obsm key name with physical location of spots/cells
            (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data
            (default 'batch' if exists on .obs, set -1 to ignore)
        num_layers (int): number of layers for decoders and encoders (default 3)
        num_neurons (int): number of neurons in each layer (default 1024)
        latent_dim (int): size of ENVI latent dimension (size 512)
        k_nearest (int): number of nearest neighbors to describe niche (default 8)
        num_covet_genes (int): number of HVGs to compute COVET
            with default (64), if -1 takes all genes
        covet_genes (list of str): manual genes to compute niche with (default [])
        num_HVG (int): number of HVGs to keep for single cell data (default 2048),
            if -1 takes all genes
        spatial_distribution (str): distribution used to describe spatial data
            (default pois, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm')
        sc_distribution (str): distribution used to describe single cell data
            (default nb, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm')
        covet_distribution (str): distance metric used for COVET matrices
            (default OT, could be 'OT', 'wish' or 'norm')
        prior_distribution (str): prior distribution for latent (default normal)
        share_dispersion (bool): whether to share dispersion parameters between
            spatial_distribution and sc_distribution (default False)
        const_dispersion (bool): if True, dispersion parameter(s) are only per gene
            rather there per gene per sample (default False)
        spatial_weight (float): coefficient for spatial expression loss in total ELBO
            (default 1.0)
        sc_weight (float): coefficient for single cell expression loss in total ELBO
            (default 1.0)
        covet_weight (float): coefficient for spatial niche loss in total ELBO
            (default 1.0)
        kl_weight (float): coefficient for latent prior loss in total ELBO (default 1.0)
        log_input (float): if larger than zero, a log is applied to input with
            pseudocount of log_input (default 0.0)
        covet_pseudocount (float): pseudocount to add to spatial_data when
            computing COVET. Only used when positive. (default 1.0)
        library_size (float or Bool) = if true, performs median library size
            if number, normalize library size to it
            if False does nothing (default False)
            var will take a per-gene average weighed by elements in anndata.var[var]
        init_scale (float): scale for VarianceScaling normalization of
            initial layer parameters (default 1.0)
        stable (float): pseudocount for rate parameter to stabilize training
            (default 1e-6)
    """

    def __init__(
        self,
        spatial_data,
        sc_data,
        spatial_key="spatial",
        batch_key=None,
        num_layers=3,
        num_neurons=1024,
        latent_dim=512,
        k_nearest=8,
        num_covet_genes=64,
        covet_genes=[],
        num_HVG=2048,
        spatial_distribution="pois",
        covet_distribution="OT",
        sc_distribution="nb",
        prior_distribution="norm",
        share_dispersion=False,
        const_dispersion=False,
        spatial_weight=1,
        sc_weight=1,
        covet_weight=1,
        kl_weight=0.3,
        log_input=0.1,
        covet_pseudocount=1,
        library_size=False,
        init_scale=0.1,
        stable=1e-6,
        covet_batch_size = 256,
        seed = 2025,
        **kwargs,
    ):
        super(ENVI, self).__init__()
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.spatial_data = spatial_data.copy()
        self.sc_data = sc_data.copy()
        self.library_size = library_size

        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.latent_dim = latent_dim

        self.spatial_distribution = spatial_distribution
        self.covet_distribution = covet_distribution
        self.sc_distribution = sc_distribution
        self.share_dispersion = share_dispersion
        self.const_dispersion = const_dispersion

        self.prior_distribution = prior_distribution

        self.spatial_weight = spatial_weight
        self.sc_weight = sc_weight
        self.covet_weight = covet_weight
        self.kl_weight = kl_weight
        self.covet_batch_size = covet_batch_size

        self.num_HVG = num_HVG
        
        if issparse(self.spatial_data.X):
            self.spatial_data.X = self.spatial_data.X.A
            
        if issparse(self.sc_data.X):
            self.sc_data.X = self.sc_data.X.A

        self.overlap_genes = np.asarray(
            np.intersect1d(self.spatial_data.var_names, self.sc_data.var_names))
        self.spatial_data = self.spatial_data[:, list(self.overlap_genes)]

        self.sc_data.layers["log"] = np.log(self.sc_data.X + 1)

        if self.num_HVG == -1:
            self.sc_data.var.highly_variable = True
        else:
            sc.pp.highly_variable_genes(
                self.sc_data,
                n_top_genes=min(self.num_HVG, self.sc_data.shape[-1]),
                layer="log")

        self.sc_data = self.sc_data[
            :,
            np.union1d(
                np.array(self.sc_data.var_names)[self.sc_data.var.highly_variable],
                self.spatial_data.var_names)
        ]

        self.non_overlap_genes = np.asarray(
            list(set(self.sc_data.var_names) - set(self.spatial_data.var_names))
        )
        self.sc_data = self.sc_data[
            :, list(self.overlap_genes) + list(self.non_overlap_genes)
        ]       ## adjust the order, making the overlapping genes at first
    
        if self.library_size:
            sc.pp.normalize_per_cell(self.spatial_data, counts_per_cell_after=np.median(self.spatial_data.X.sum(axis=1)))
            sc.pp.normalize_per_cell(self.sc_data, counts_per_cell_after=np.median(self.sc_data.X.sum(axis=1)))
            
        self.k_nearest = k_nearest
        self.spatial_key = spatial_key

        if batch_key is not None and batch_key in self.spatial_data.obs.columns:
            self.batch_key = batch_key
        else:
            self.batch_key = -1
        
        self.num_covet_genes = min(num_covet_genes, self.spatial_data.shape[-1])
        self.covet_genes = covet_genes
        self.covet_pseudocount = covet_pseudocount

        self.spatial_data = self.spatial_data.copy()
        (
            self.spatial_data.obsm["COVET"],
            self.spatial_data.obsm["COVET_SQRT"],
            self.covet_genes,
        ) = get_covet(
            self.spatial_data,
            self.k_nearest,
            self.num_covet_genes,
            self.covet_genes,
            self.covet_distribution,
            spatial_key=self.spatial_key,
            batch_key=self.batch_key,
            covet_pseudocount=self.covet_pseudocount,
            covet_batch_size = self.covet_batch_size
        )

        self.n_overlap_genes = self.overlap_genes.shape[0]
        self.n_covet_genes = self.spatial_data.obsm["COVET_SQRT"].shape[-1]
        self.n_whole_transcriptome_genes = self.sc_data.shape[-1]

        self.log_spatial = False
        self.log_sc = False

        self.log_input = log_input
        self.stable = stable
        self.init_scale = init_scale
        
        self.data_scale = np.abs(spatial_data.X).mean() if spatial_data is not None else 1.0
        
        std_layers = np.sqrt(self.init_scale / self.num_neurons) / self.data_scale
        std_encoder = np.sqrt(self.init_scale / self.num_neurons)
        std_output_covet = np.sqrt(self.init_scale / self.num_neurons)
        std_output_expr = np.sqrt(self.init_scale / self.overlap_genes.shape[0])
        
        self.encoder_layers = nn.ModuleList()
        self.decoder_expression_layers = nn.ModuleList()
        self.decoder_covet_layers = nn.ModuleList()

        for i in range(num_layers - 1):
            layer = nn.Linear(self.num_neurons if i > 0 else self.n_overlap_genes+2, num_neurons)
            init_weights(layer, std=std_layers)
            self.encoder_layers.append(layer)

        # Encoder output layer (2 * latent_dim)
        encoder_output = nn.Linear(self.num_neurons, 2 * latent_dim)
        init_weights(encoder_output, std=std_encoder)
        self.encoder_layers.append(encoder_output)

        # Decoder (expression) hidden layers
        for i in range(num_layers - 1):
            layer = nn.Linear(self.num_neurons if i > 0 else self.latent_dim + 2, self.num_neurons)
            init_weights(layer, std=std_layers)
            self.decoder_expression_layers.append(layer)

        self.decoder_expression_layers.append(
            ENVIOutputLayer(
                input_dim=num_neurons,
                output_dim=self.n_whole_transcriptome_genes,
                spatial_distribution=spatial_distribution,
                sc_distribution=sc_distribution,
                share_dispersion=share_dispersion,
                const_dispersion=const_dispersion,
                kernel_init=std_output_expr,
                bias_init=std_output_expr,
                name="decoder_expression_output"
            )
        )

        # Decoder (covet) hidden layers
        for i in range(num_layers - 1):
            layer = nn.Linear(self.num_neurons if i > 0 else self.latent_dim, num_neurons)
            init_weights(layer, std=std_layers)
            self.decoder_covet_layers.append(layer)

        # Decoder (covet) output layer
        decoder_covet_out = nn.Linear(
            num_neurons,
            int(self.n_covet_genes * (self.n_covet_genes + 1) / 2)
        )
        init_weights(decoder_covet_out, std=std_output_covet)
        self.decoder_covet_layers.append(decoder_covet_out)
          
        
    def encode_nn(self, input):
        """
        Encoder forward pass

        Args:
            input (array): input to encoder NN
                (size of #genes in spatial data + confounder)
        Returns:
            output (array): NN output
        """

        output = input
        for i in range(self.num_layers - 1):
            residual = output if (i > 0) else 0
            output = self.encoder_layers[i](output) + residual
            # output = F.leaky_relu(output)
            output = F.relu(output)
        return self.encoder_layers[-1](output)


    def decode_expression_nn(self, input):
        """
        Expression decoder forward pass

        Args:
            input (array): input to expression decoder NN
                (size of latent dimension + confounder)

        Returns:
            output (array): NN output
        """

        output = input
        for i in range(self.num_layers - 1):
            output = self.decoder_expression_layers[i](output) + (
                output if (i > 0 ) else 0
            )
            # output = F.leaky_relu(output)
            output = F.relu(output)
        return output

    def decode_cov_nn(self, input):
        """
        Covariance (niche) decoder forward pass

        Args:
            input (array): input to niche decoder NN
                (size of latent dimension + confounder)

        Returns:
            output (array): NN output
        """
        output = input
        for i in range(self.num_layers - 1):
            output = self.decoder_covet_layers[i](output) + (
                output if (i > 0 ) else 0
            )
            # output = F.leaky_relu(output)
            output = F.relu(output)
        return self.decoder_covet_layers[-1](output)
    
    def encode(self, x, mode="sc"):
        """
        Appends a confounding variable (one-hot encoded) to the input and generates an encoding.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_genes]
            mode (str): 'sc' for single cell or 'spatial' for spatial data

        Returns:
            Tuple[Tensor, Tensor]: mean and log_std tensors from the encoder output
        """

        # Determine which confounder type (0 = spatial, 1 = sc)
        confounder = 0 if mode == "spatial" else 1

        # Optional log transform
        if self.log_input > 0:
            x = torch.log(x + self.log_input)

        confounder_one_hot = F.one_hot(
            torch.full((x.size(0),), confounder, dtype=torch.long),
            num_classes=2
        ).type_as(x)  # match dtype to input tensor

        x_confounder = torch.cat([x, confounder_one_hot], dim=-1)
        encoded = self.encode_nn(x_confounder)  # shape: [batch_size, 2 * latent_dim]

        # Split into mean and log_std
        mean, log_std = torch.chunk(encoded, chunks=2, dim=1)
        return mean, log_std
    
    def expression_decode(self, x, mode="sc"):
        """
        Appends confounding variable to latent input and generates distribution parameterizations.

        Args:
            x (Tensor): Input tensor of shape [batch_size, latent_dim]
            mode (str): 'sc' or 'spatial'

        Returns:
            Tensor(s): Parameters of the decoded output distribution
        """

        # Set confounder based on mode
        confounder = 0 if mode == "spatial" else 1

        # One-hot encode the confounder
        confounder_one_hot = F.one_hot(
            torch.full((x.size(0),), confounder, dtype=torch.long),
            num_classes=2
        ).type_as(x)

        # Concatenate latent + confounder
        x_confounder = torch.cat([x, confounder_one_hot], dim=-1)

        # Pass through decoder MLP
        decoder_output = self.decode_expression_nn(x_confounder)

        dist = getattr(self, f"{mode}_distribution")

        # Pass through final decoder output layer (ENVIOutputLayer)
        if dist == "zinb":
            output_r, output_p, output_d = self.decoder_expression_layers[-1](decoder_output, mode)
            return (
                F.softplus(output_r) + self.stable,
                output_p,
                torch.sigmoid(0.01 * output_d - 2),
            )

        elif dist == "nb":
            output_r, output_p = self.decoder_expression_layers[-1](decoder_output, mode)
            return F.softplus(output_r) + self.stable, output_p

        elif dist == "pois":
            output_l = self.decoder_expression_layers[-1](decoder_output, mode)
            return F.softplus(output_l) + self.stable

        elif dist == "full_norm":
            output_mu, output_logstd = self.decoder_expression_layers[-1](decoder_output, mode)
            return output_mu, output_logstd

        elif dist == "norm":
            output_mu = self.decoder_expression_layers[-1](decoder_output, mode)
            return output_mu

        else:
            raise ValueError(f"Unsupported distribution type: {dist}")
        
    def covet_decode(self, x):
        """
        Generates output for niche (COVET) decoder.

        Args:
            x (Tensor): Input tensor of shape [batch_size, latent_dim]

        Returns:
            Tensor: A reconstructed triangular matrix depending on the distribution type.
        """

        DecOut = self.decode_cov_nn(x)  # shape: [B, D*(D+1)/2]
        TriMat = fill_triangular(DecOut)  # shape: [B, D, D]

        if self.covet_distribution == "norm":
            TriMat_T = TriMat.transpose(1, 2)
            return 0.5 * (TriMat + TriMat_T)
        elif self.covet_distribution == "OT":
            return TriMat @ TriMat.transpose(1, 2)
        else:
            raise ValueError(f"Unsupported COVET distribution: {self.covet_distribution}")
        

    def encoder_mean(self, mean, logstd):
        """
        Computes the posterior mean depending on the prior distribution.

        Args:
            mean (Tensor): latent mean
            logstd (Tensor): latent log-std

        Returns:
            Tensor: Posterior mean
        """
        if self.prior_distribution == "norm":
            return mean
        elif self.prior_distribution == "log_norm":
            # E[logNormal] = exp(mean + var/2)
            return torch.exp(mean + torch.exp(logstd).pow(2) / 2)
        else:
            raise ValueError(f"Unsupported prior: {self.prior_distribution}")
        
    def reparameterize(self, mean, logstd):
        """
        Reparameterization trick to sample from q(z|x)

        Args:
            mean (Tensor): latent mean
            logstd (Tensor): latent log-std

        Returns:
            Tensor: sample z
        """
        eps = torch.randn_like(mean)
        z = eps * torch.exp(logstd) + mean
        if self.prior_distribution == "norm":
            return z
        elif self.prior_distribution == "log_norm":
            return torch.exp(z)
        else:
            raise ValueError(f"Unsupported prior: {self.prior_distribution}")
        
        
    def compute_loss(self, spatial_sample, cov_sample, sc_sample):
        """
        Computes ENVI likelihoods and KL divergence.

        Args:
            spatial_sample (Tensor): spatial expression data
            cov_sample (Tensor): COVET data
            sc_sample (Tensor): single-cell expression data

        Returns:
            Tuple: (spatial_likelihood, covet_likelihood, sc_likelihood, kl)
        """

        # Encode inputs
        mean_spatial, logstd_spatial = self.encode(spatial_sample[:, :self.n_overlap_genes], mode="spatial")
        mean_sc, logstd_sc = self.encode(sc_sample[:, :self.n_overlap_genes], mode="sc")

        z_spatial = self.reparameterize(mean_spatial, logstd_spatial)
        z_sc = self.reparameterize(mean_sc, logstd_sc)

        ### Spatial likelihood ###
        if self.spatial_distribution == "zinb":
            r, p, d = self.expression_decode(z_spatial, mode="spatial")
            spatial_likelihood = log_zinb_pdf(
                spatial_sample, r[:, :spatial_sample.shape[-1]], p[:, :spatial_sample.shape[-1]], d[:, :spatial_sample.shape[-1]]
            ).mean(dim=0)

        elif self.spatial_distribution == "nb":
            r, p = self.expression_decode(z_spatial, mode="spatial")
            spatial_likelihood = log_nb_pdf(
                spatial_sample, r[:, :spatial_sample.shape[-1]], p[:, :spatial_sample.shape[-1]]
            ).mean(dim=0)

        elif self.spatial_distribution == "pois":
            l = self.expression_decode(z_spatial, mode="spatial")
            spatial_likelihood = log_pos_pdf(
                spatial_sample, l[:, :spatial_sample.shape[-1]]
            ).mean(dim=0)

        ### Single-cell likelihood ###
        if self.sc_distribution == "zinb":
            r, p, d = self.expression_decode(z_sc, mode="sc")
            sc_likelihood = log_zinb_pdf(sc_sample, r, p, d).mean(dim=0)

        elif self.sc_distribution == "nb":
            r, p = self.expression_decode(z_sc, mode="sc")
            sc_likelihood = log_nb_pdf(sc_sample, r, p).mean(dim=0)

        elif self.sc_distribution == "pois":
            l = self.expression_decode(z_sc, mode="sc")
            sc_likelihood = log_pos_pdf(sc_sample, l).mean(dim=0)

        ### Covet likelihood ###
        cov_mu = self.covet_decode(z_spatial)

        if self.covet_distribution == "norm":
            cov_mu = cov_mu.reshape(spatial_sample.shape[0], -1)
            covet_likelihood = log_normal_pdf(
                cov_sample.reshape(cov_sample.shape[0], -1),
                cov_mu,
                torch.zeros_like(cov_mu)
            ).mean(dim=0)

        elif self.covet_distribution == "OT":
            covet_likelihood = ot_distance(cov_sample, cov_mu).mean(dim=0)

        ### KL divergence ###
        kl_spatial = normal_kl(mean_spatial, logstd_spatial).mean(dim=0)
        kl_sc = normal_kl(mean_sc, logstd_sc).mean(dim=0)

        kl = 0.5 * kl_spatial + 0.5 * kl_sc
        return spatial_likelihood, covet_likelihood, sc_likelihood, kl
    
    def get_covet_mean(self, covet):
        """
        Untransforms the COVET matrix based on the distribution used.

        Args:
            covet (Tensor): transformed COVET matrix

        Returns:
            Tensor: untransformed (original scale) COVET matrix
        """
        if self.covet_distribution == "OT":
            return covet @ covet
        else:
            return covet

    def get_mean_sample(self, decode, mode="spatial"):
        """
        Computes the mean of the expression distribution from decoded parameters.

        Args:
            decode (Tensor or list of Tensors): distribution parameters
            mode (str): 'spatial' or 'sc'

        Returns:
            Tensor: mean of the expression distribution
        """
        dist = getattr(self, f"{mode}_distribution")

        if dist == "zinb":
            r, p, d = decode
            return r * torch.exp(p) * (1 - d)

        elif dist == "nb":
            r, p = decode
            return r * torch.exp(p)

        elif dist == "pois":
            return decode  # Poisson mean is lambda

        elif dist == "full_norm":
            mu, _ = decode
            return mu

        elif dist == "norm":
            return decode  # Only mean is returned

        else:
            raise ValueError(f"Unsupported distribution: {dist}")
        

    def latent_rep(self, num_div=16, data=None, mode=None):
        """
        Compute latent embeddings for spatial and single-cell data.

        Args:
            num_div (int): number of chunks for memory-safe forward pass
            data (AnnData or None): optional new data to embed
            mode (str or None): 'spatial' or 'sc' when data is provided

        Returns:
            If data is None, updates `.obsm["envi_latent"]` for spatial_data and sc_data
            If data is provided, returns the latent embedding (numpy array)
        """

        def batch_encode(x, mode):
            # Split data into chunks and run through encoder
            chunks = np.array_split(x, num_div, axis=0)
            latent = [
                self.encode(torch.from_numpy(chunk).to(self.device).float(), mode=mode)[0].detach().cpu().numpy()
                for chunk in chunks
            ]
            return np.concatenate(latent, axis=0)

        if data is None:
            # --- Spatial Data ---
            spatial_input = self.spatial_data.X.astype(np.float32)
            self.spatial_data.obsm["envi_latent"] = batch_encode(spatial_input, mode="spatial")

            # --- Single Cell Data ---
            sc_input = self.sc_data[:, self.spatial_data.var_names].X.astype(np.float32)
            self.sc_data = self.sc_data.copy()
            self.sc_data.obsm["envi_latent"] = batch_encode(sc_input, mode="sc")

        else:
            # Validate gene overlap
            if not set(self.spatial_data.var_names).issubset(set(data.var_names)):
                print(f"({mode}) Data does not contain trained genes.")
                return -1

            # Subset to trained genes
            if mode == "spatial":
                ref_data = self.spatial_data
            else:
                ref_data = self.sc_data

            data = data[:, ref_data.var_names].copy()

            # Compute and return latent representation
            input_array = data.X.astype(np.float32)
            return batch_encode(input_array, mode=mode)

    def train(
        self,
        lr=1e-4,
        batch_size=512,
        # epochs=2**14,
        epochs = 10000,
        verbose=64,
        lr_schedule=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed = 2025
    ):
        """
        PyTorch training loop for ENVI model.

        Args:
            lr (float): Initial learning rate
            batch_size (int): Batch size
            epochs (int): Number of training steps
            verbose (int): Logging frequency (every N steps)
            lr_schedule (bool): If True, reduce LR by 10x at 75% of training
            device (str): 'cuda' or 'cpu'
        """

        print(f"Training ENVI on {device}")
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed_all(seed)
        
        self.device = device
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Convert data to PyTorch tensors
        spatial_data = torch.tensor(self.spatial_data.X, dtype=torch.float32, device=device)
        cov_data = torch.tensor(self.spatial_data.obsm["COVET_SQRT"], dtype=torch.float32, device=device)
        sc_data = torch.tensor(self.sc_data.X, dtype=torch.float32, device=device)

        tq = trange(epochs, desc="")

        for step in tq:
            if lr_schedule and step == int(epochs * 0.75):
                for g in self.optimizer.param_groups:
                    g["lr"] *= 0.1

            spatial_idx = torch.randint(0, len(spatial_data), (min(batch_size, len(spatial_data)),))
            sc_idx = torch.randint(0, len(sc_data), (min(batch_size, len(sc_data)),))

            spatial_sample = spatial_data[spatial_idx]
            cov_sample = cov_data[spatial_idx]
            sc_sample = sc_data[sc_idx]

            # Forward + backward pass
            self.optimizer.zero_grad()
            spatial_likelihood, covet_likelihood, sc_likelihood, kl = self.compute_loss(
                spatial_sample, cov_sample, sc_sample
            )

            loss = (
                -self.spatial_weight * spatial_likelihood
                - self.sc_weight * sc_likelihood
                - self.covet_weight * covet_likelihood
                + 2 * self.kl_weight * kl
            )

            # Check for NaNs
            if torch.isnan(loss).any():
                print("NaN encountered in loss. Skipping step.")
                continue

            loss.backward()
            self.optimizer.step()

            # Print training progress
            if verbose > 0 and step % verbose == 0:
                msg = (
                    f"Trn: spatial Loss: {spatial_likelihood.item():.5f}, "
                    f"SC Loss: {sc_likelihood.item():.5f}, "
                    f"Cov Loss: {covet_likelihood.item():.5f}, "
                    f"KL Loss: {kl.item():.5f}"
                )
                tq.set_description(msg, refresh=True)

        print("Finished initializing neighboring gene-gene covariance estimation.")
        self.latent_rep()
        
        del self.spatial_data.obsm["COVET_SQRT"]
        
        
    def infer_covet(self, num_div=16, data=None):
        """
        Predict COVET matrices for single-cell data.

        Args:
            num_div (int): number of batches for GPU efficiency
            data (AnnData): optional new dataset to use instead of ENVI.sc_data

        Returns:
            None (saves to .obsm) or tuple (covet_sqrt, covet)
        """

        def decode_covet_chunks(latent):
            chunks = np.array_split(latent, num_div, axis=0)
            decoded = [
                self.covet_decode(torch.from_numpy(chunk).to(self.device).float()).detach().cpu().numpy()
                for chunk in chunks
            ]
            return np.concatenate(decoded, axis=0)

        if data is None:
            latent = self.sc_data.obsm["envi_latent"]
        else:
            latent = self.latent_rep(data=data, mode="sc")

        covet_sqrt = decode_covet_chunks(latent)

        # Matrix square root → full matrix: A @ A.T
        covet = np.matmul(covet_sqrt, np.transpose(covet_sqrt, (0, 2, 1)))
        del covet_sqrt

        if data is None:
            # self.sc_data.obsm["COVET_SQRT"] = covet_sqrt
            self.sc_data.obsm["COVET"] = covet
        else:
            return covet
        