import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Self-Attention block (SAGAN style) for spatial feature maps.
    Enables the network to model long-range spatial dependencies.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.size()
        
        # Query projection: [B, C//8, H*W]
        q = self.query(x).view(batch, -1, height * width)
        # Key projection: [B, C//8, H*W]
        k = self.key(x).view(batch, -1, height * width)
        # Value projection: [B, C, H*W]
        v = self.value(x).view(batch, -1, height * width)
        
        # Attention map: [B, H*W, H*W]
        attn = torch.bmm(q.permute(0, 2, 1), k)
        attn = self.softmax(attn)
        
        # Out projection: [B, C, H*W]
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        
        # Residual connection
        return x + self.gamma * out


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    Normalizes feature maps dynamically by prioritizing important feature channels.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ResNetBlock(nn.Module):
    """
    Standard CycleGAN ResNet block with Squeeze-and-Excitation (SE) channel attention.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            SqueezeExcitation(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    """
    SOTA ResNet-based Generator with 9 ResNet blocks, Channel Attention (SE), 
    and Self-Attention (SA) in the middle of the bottleneck.
    """
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=9):
        super().__init__()
        
        # 1. Downsampling (Encoder)
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, num_features, kernel_size=7, stride=1, bias=False),
            nn.InstanceNorm2d(num_features, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(num_features, num_features * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features * 2, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features * 4, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # 2. ResNet bottleneck blocks
        blocks = [ResNetBlock(num_features * 4) for _ in range(num_blocks)]
        self.bottleneck = nn.Sequential(*blocks)
        
        # 3. Upsampling (Decoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(num_features * 2, affine=True),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(num_features, affine=True),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(num_features, out_channels, kernel_size=7, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

    def forward_with_features(self, x, encode_only=False):
        features = []
        
        # Raw input is features[0]
        features.append(x)
        
        # Run encoder layer by layer
        feat = x
        for idx, sub_layer in enumerate(self.encoder):
            feat = sub_layer(feat)
            # Save features after ReLU downsampling stages:
            # - idx=3 (after first conv block)
            # - idx=6 (after second conv block)
            # - idx=9 (after third conv block)
            if idx in [3, 6, 9]:
                features.append(feat)
                
        # Run bottleneck block by block
        for idx, block in enumerate(self.bottleneck):
            feat = block(feat)
            # Save features after first and middle bottleneck blocks:
            # - idx=0 (first bottleneck block)
            # - idx=4 (middle bottleneck block)
            if idx in [0, 4]:
                features.append(feat)
                
        if encode_only:
            return features
            
        out = self.decoder(feat)
        return out, features


class PatchSampleF(nn.Module):
    """
    Projector and patch sampler for Contrastive Learning (CUT).
    Extracts features at specified spatial positions and projects them to a shared embedding space.
    """
    def __init__(self, in_channels_list, projection_dim=256):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)
            ) for in_channels in in_channels_list
        ])

    def forward(self, feats, num_patches=256, patch_ids=None):
        return_feats = []
        return_patch_ids = []
        
        for i, feat in enumerate(feats):
            B, C, H, W = feat.shape
            feat_flat = feat.view(B, C, H * W)
            
            if patch_ids is None:
                if H * W < num_patches:
                    # Sample with replacement if spatial resolution is very small
                    import torch
                    p_ids = torch.randint(0, H * W, (num_patches,), device=feat.device)
                else:
                    import torch
                    p_ids = torch.randperm(H * W, device=feat.device)[:num_patches]
                return_patch_ids.append(p_ids)
            else:
                p_ids = patch_ids[i]
                
            feat_sampled = feat_flat[:, :, p_ids] # [B, C, num_patches]
            feat_sampled = feat_sampled.permute(0, 2, 1) # [B, num_patches, C]
            
            # Run through projection layer
            feat_projected = self.mlps[i](feat_sampled) # [B, num_patches, projection_dim]
            return_feats.append(feat_projected)
            
        return return_feats, return_patch_ids


class SpectralPatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator stabilized with Spectral Normalization and Instance Normalization.
    """
    def __init__(self, in_channels=3, num_features=64):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(num_features, num_features * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(num_features * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(num_features * 2, num_features * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(num_features * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(num_features * 4, num_features * 8, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(num_features * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(num_features * 8, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale PatchGAN Discriminator.
    Evaluates inputs at the original resolution (128x128) and a 2x downsampled resolution (64x64).
    """
    def __init__(self, in_channels=3, num_features=64):
        super().__init__()
        self.disc_scale1 = SpectralPatchDiscriminator(in_channels, num_features)
        self.disc_scale2 = SpectralPatchDiscriminator(in_channels, num_features)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        # Scale 1 (original resolution)
        out1 = self.disc_scale1(x)
        # Scale 2 (downsampled resolution)
        x_down = self.downsample(x)
        out2 = self.disc_scale2(x_down)
        return out1, out2
