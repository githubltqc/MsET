import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import escnn.gspaces as gspaces
from escnn import nn as enn
import torch.utils.checkpoint as checkpoint


# ---------------------------- MultiScaleFeatureExtractor ----------------------------
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_type, output_type, r2_act):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.r2_act = r2_act  # Equivariant space
        self.repr_size = self.r2_act.regular_repr.size

        # Define multi-scale convolutions
        self.conv3x3 = enn.R2Conv(
            input_type, output_type, kernel_size=3, padding=1, bias=False
        )
        self.conv5x5 = enn.R2Conv(
            input_type, output_type, kernel_size=5, padding=2, bias=False
        )
        self.conv7x7 = enn.R2Conv(
            input_type, output_type, kernel_size=7, padding=3, bias=False
        )

        # Pyramid pooling layers
        self.pool1 = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.pool2 = nn.AdaptiveAvgPool2d(2)  # 2x2 pooling
        self.pool4 = nn.AdaptiveAvgPool2d(4)  # 4x4 pooling

        # 1x1 convolution for dimensionality reduction after pooling
        self.pool_conv = nn.Conv2d(
            output_type.size,  # Correct in_channels
            output_type.size,  # Correct out_channels
            kernel_size=1,
            bias=False,
        )

        # Final fusion layer
        # Expects 6 * output_type.size // repr_size fields
        self.fuse_conv = enn.R2Conv(
            enn.FieldType(
                self.r2_act,
                6 * output_type.size // self.repr_size * [self.r2_act.regular_repr],
            ),
            output_type,
            kernel_size=1,
            bias=False,
        )


    def forward(self, x):
        # Multi-scale convolution features
        x3 = F.relu(self.conv3x3(x).tensor)
        x5 = F.relu(self.conv5x5(x).tensor)
        x7 = F.relu(self.conv7x7(x).tensor)

        # Wrap features into GeometricTensor
        x3 = enn.GeometricTensor(x3, self.conv3x3.out_type)
        x5 = enn.GeometricTensor(x5, self.conv5x5.out_type)
        x7 = enn.GeometricTensor(x7, self.conv7x7.out_type)

        # Pyramid pooling features
        x_pool1 = self.pool1(x.tensor)
        x_pool2 = self.pool2(x.tensor)
        x_pool4 = self.pool4(x.tensor)

        # Pooling results passed through 1x1 convolution
        x_pool1 = F.relu(self.pool_conv(x_pool1))
        x_pool2 = F.relu(self.pool_conv(x_pool2))
        x_pool4 = F.relu(self.pool_conv(x_pool4))

        # Upsample pooled features to match input size
        x_pool1 = F.interpolate(
            x_pool1, size=x.tensor.shape[2:], mode="bilinear", align_corners=False
        )
        x_pool2 = F.interpolate(
            x_pool2, size=x.tensor.shape[2:], mode="bilinear", align_corners=False
        )
        x_pool4 = F.interpolate(
            x_pool4, size=x.tensor.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate all features along the channel dimension
        x_combined = torch.cat(
            [x3.tensor, x5.tensor, x7.tensor, x_pool1, x_pool2, x_pool4], dim=1
        )

        # Determine the number of fields based on representation size
        n_fields = x_combined.shape[1] // self.repr_size

        # Ensure that the number of channels is divisible by the representation size
        assert (
            x_combined.shape[1] == n_fields * self.repr_size
        ), f"Number of channels ({x_combined.shape[1]}) is not divisible by the representation size ({self.repr_size})."

        # Wrap combined features into GeometricTensor with correct FieldType
        x_combined = enn.GeometricTensor(
            x_combined,
            enn.FieldType(self.r2_act, n_fields * [self.r2_act.regular_repr]),
        )

        # Final fusion
        x_fused = self.fuse_conv(x_combined)

        return x_fused


# ---------------------------- Transformer Components ----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SpatialEquivariantAttention(nn.Module):
    def __init__(self, d_model=64, hidden_dim=32):
        super(SpatialEquivariantAttention, self).__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(d_model)

        # Define Q, K, V linear transformations
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)

        # Define MLPs for rotation, translation, reflection
        self.rot_mlp = MLP(input_dim=3, hidden_dim=hidden_dim, output_dim=1)
        self.trans_mlp = MLP(input_dim=2, hidden_dim=hidden_dim, output_dim=1)
        self.reflect_mlp = MLP(input_dim=4, hidden_dim=hidden_dim, output_dim=1)

    def forward(self, x, coords):
        batch_size, seq_len, d_model = x.size()

        # Compute Q, K, V
        Q = self.q_linear(x)  # (batch, seq_len, d_model)
        K = self.k_linear(x)  # (batch, seq_len, d_model)
        V = self.v_linear(x)  # (batch, seq_len, d_model)

        # Compute scaled dot-product attention scores
        scores = (
            torch.bmm(Q, K.transpose(1, 2)) / self.scale
        )  # (batch, seq_len, seq_len)

        # Compute relative positions
        coords_i = coords.unsqueeze(2)  # (batch, seq_len, 1, 2)
        coords_j = coords.unsqueeze(1)  # (batch, 1, seq_len, 2)
        relative_pos = coords_j - coords_i  # (batch, seq_len, seq_len, 2)

        dx = relative_pos[..., 0]  # (batch, seq_len, seq_len)
        dy = relative_pos[..., 1]  # (batch, seq_len, seq_len)

        # Compute distance and angle
        distance = torch.sqrt(dx**2 + dy**2 + 1e-8)  # (batch, seq_len, seq_len)
        theta = torch.atan2(dy, dx)  # (batch, seq_len, seq_len)

        # Compute sin(theta) and cos(theta)
        sin_theta = torch.sin(theta)  # (batch, seq_len, seq_len)
        cos_theta = torch.cos(theta)  # (batch, seq_len, seq_len)

        # Rotation attention embedding
        rot_input = torch.stack(
            [distance, sin_theta, cos_theta], dim=-1
        )  # (batch, seq_len, seq_len, 3)
        psi_rot = self.rot_mlp(rot_input).squeeze(-1)  # (batch, seq_len, seq_len)

        # Translation attention embedding
        trans_input = torch.stack([dx, dy], dim=-1)  # (batch, seq_len, seq_len, 2)
        psi_trans = self.trans_mlp(trans_input).squeeze(-1)  # (batch, seq_len, seq_len)

        # Reflection attention embedding
        reflect_input = torch.stack(
            [dx, dy, -dx, -dy], dim=-1
        )  # (batch, seq_len, seq_len, 4)
        psi_reflect = self.reflect_mlp(reflect_input).squeeze(
            -1
        )  # (batch, seq_len, seq_len)

        # Add geometric embeddings to attention scores
        scores = scores + psi_rot + psi_trans + psi_reflect  # (batch, seq_len, seq_len)

        # Apply softmax
        attn = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # Attention output
        out = torch.bmm(attn, V)  # (batch, seq_len, d_model)

        return out


class SpatialEquivariantTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=64, hidden_dim=32, ffn_dim=128, dropout=0.1):
        super(SpatialEquivariantTransformerEncoderLayer, self).__init__()
        self.attention = SpatialEquivariantAttention(d_model, hidden_dim)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, d_model)
        )
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, coords):
        # Self-Attention sublayer with checkpointing
        def attn_forward(x, coords):
            return self.attention(x, coords)

        attn_out = checkpoint.checkpoint(attn_forward, x, coords)
        x = self.attn_layer_norm(x + self.dropout(attn_out))  # Residual and norm

        # Feed-forward sublayer with checkpointing
        def ffn_forward(x):
            return self.ffn(x)

        ffn_out = checkpoint.checkpoint(ffn_forward, x)
        x = self.ffn_layer_norm(x + self.dropout(ffn_out))  # Residual and norm

        return x


class SpatialEquivariantTransformerEncoder(nn.Module):
    def __init__(
        self, num_layers=2, d_model=64, hidden_dim=32, ffn_dim=128, dropout=0.1
    ):
        super(SpatialEquivariantTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                SpatialEquivariantTransformerEncoderLayer(
                    d_model=d_model,
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, coords):
        for layer in self.layers:
            x = layer(x, coords)
        x = self.layer_norm(x)
        return x


# ---------------------------- Coordinate Generation ----------------------------
def generate_normalized_coordinates(batch_size, seq_len, device="cpu"):
    coord_dim = 2  # 2D coordinates
    grid_size = math.ceil(math.sqrt(seq_len))  # Nearest grid size
    total_points = grid_size**2  # Total points in grid

    # Generate grid coordinates with 'ij' indexing to suppress warning
    x_coords = torch.linspace(0, 1, steps=grid_size, device=device)
    y_coords = torch.linspace(0, 1, steps=grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")  # 'ij' indexing
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    if seq_len <= total_points:
        coords = torch.stack([grid_x, grid_y], dim=1)[:seq_len]  # (seq_len,2)
    else:
        # If seq_len > total_points, repeat grid to reach seq_len
        repeats = math.ceil(seq_len / total_points)
        coords = torch.stack([grid_x, grid_y], dim=1).repeat(repeats, 1)[:seq_len]

    # Expand to batch
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, seq_len, 2)
    return coords


# ---------------------------- Combined Neural Network ----------------------------
class CombinedNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_classes,
        num_transformer_layers=2,
        d_model=64,  # Match conv3_type channels
        hidden_dim=32,
        ffn_dim=128,
        dropout=0.1,
    ):
        super(CombinedNN, self).__init__()

        # 定义一个包含旋转、翻转和平移的综合群（P4M群）
        self.r2_act = gspaces.flipRot2dOnR2(N=4)
        self.repr_size = self.r2_act.regular_repr.size  # 通常为8

        # Define the initial equivariant convolution layer
        self.input_type = enn.FieldType(
            self.r2_act, in_channels * [self.r2_act.trivial_repr]
        )
        self.conv1_type = enn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])
        self.conv1 = enn.R2Conv(
            self.input_type, self.conv1_type, kernel_size=3, padding=1, bias=False
        )

        # MultiScaleFeatureExtractor after conv1
        self.msfe1 = MultiScaleFeatureExtractor(
            input_type=self.conv1_type, output_type=self.conv1_type, r2_act=self.r2_act
        )

        # Second equivariant convolution
        self.conv2_type = enn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr])
        self.conv2 = enn.R2Conv(
            self.conv1_type, self.conv2_type, kernel_size=3, padding=1, bias=False
        )
        # MultiScaleFeatureExtractor after conv2
        self.msfe2 = MultiScaleFeatureExtractor(
            input_type=self.conv2_type, output_type=self.conv2_type, r2_act=self.r2_act
        )

        # Third equivariant convolution
        self.conv3_type = enn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
        self.conv3 = enn.R2Conv(
            self.conv2_type, self.conv3_type, kernel_size=3, padding=1, bias=False
        )
        # MultiScaleFeatureExtractor after conv3
        self.msfe3 = MultiScaleFeatureExtractor(
            input_type=self.conv3_type, output_type=self.conv3_type, r2_act=self.r2_act
        )

        # GroupPooling to make features rotation invariant
        self.output_type = enn.FieldType(self.r2_act, 64 * [self.r2_act.trivial_repr])
        self.gpool = enn.GroupPooling(self.conv3_type)

        # Define Transformer Encoder
        self.transformer_encoder = SpatialEquivariantTransformerEncoder(
            num_layers=num_transformer_layers,
            d_model=d_model,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # Final fully connected layer
        self.fc = nn.Linear(d_model, out_classes)

    def forward(self, x):
        # Wrap input tensor into GeometricTensor
        x_geometric = enn.GeometricTensor(
            x, self.input_type
        )  # (batch, in_channels, H, W)

        # --- Conv1 + ReLU + msfe1 ---
        x = self.conv1(x_geometric)  # (batch, 16 fields, H, W)
        x_tensor = F.relu(x.tensor)
        x = enn.GeometricTensor(x_tensor, self.conv1_type)
        x = self.msfe1(x)

        # --- Conv2 + ReLU + msfe2 ---
        x = self.conv2(x)  # (batch, 32 fields, H, W)
        x_tensor = F.relu(x.tensor)
        x = enn.GeometricTensor(x_tensor, self.conv2_type)
        x = self.msfe2(x)

        # --- Conv3 + ReLU + msfe3 ---
        x = self.conv3(x)  # (batch, 64 fields, H, W)
        x_tensor = F.relu(x.tensor)
        x = enn.GeometricTensor(x_tensor, self.conv3_type)
        x = self.msfe3(x)

        # Apply group pooling
        x_pooled = self.gpool(x)  # (batch, 64 trivial fields, H, W)

        # Reshape for transformer: (batch, seq_len, d_model)
        batch_size, channels, height, width = x_pooled.tensor.size()
        seq_len = height * width
        x_flat = x_pooled.tensor.view(batch_size, channels, seq_len).permute(
            0, 2, 1
        )  # (batch, seq_len, 64)

        # Generate normalized coordinates
        coords = generate_normalized_coordinates(
            batch_size, seq_len, device=x_flat.device
        )

        # Apply Transformer Encoder
        x_encoded = self.transformer_encoder(
            x_flat, coords
        )  # (batch, seq_len, d_model=64)

        # Aggregate transformer outputs, e.g., mean pooling
        x_pooled_encoded = x_encoded.mean(dim=1)  # (batch, d_model=64)

        # Apply final fully connected layer
        x_out = self.fc(x_pooled_encoded)  # (batch, out_classes)

        return x_out


# ---------------------------- Testing the Combined Network ----------------------------
if __name__ == "__main__":
    import warnings

    # Suppress specific UserWarnings from e2cnn related to deprecation
    warnings.filterwarnings("ignore", category=UserWarning, module="e2cnn")

    # Create the combined model
    model = CombinedNN(in_channels=21, out_classes=16)

    # Test the model with random inputs
    x = torch.randn(8, 21, 15, 15)  # Batch of 8, 21 channels, 15x15 spatial dimensions
    y = model(x)  # Output shape should be (8, 16)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
