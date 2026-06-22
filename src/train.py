import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from src.utils import save_training_samples, save_test_samples

class CycleGANTrainer:
    """
    Trainer class for Human-Anime CycleGAN training in PyTorch.
    Manages training steps, testing steps, learning rate scheduling, mixed-precision training,
    and saving of training/testing samples and model checkpoints.
    Supports Multi-Scale Discriminators.
    """
    def __init__(
        self,
        G_XtoY,
        G_YtoX,
        Di_X,
        Di_Y,
        optimizer_G_XtoY,
        optimizer_G_YtoX,
        optimizer_Di_X,
        optimizer_Di_Y,
        scheduler_G_XtoY=None,
        scheduler_G_YtoX=None,
        scheduler_Di_X=None,
        scheduler_Di_Y=None,
        use_amp=True,
        device='cuda'
    ):
        self.G_XtoY = G_XtoY.to(device)
        self.G_YtoX = G_YtoX.to(device)
        self.Di_X = Di_X.to(device)
        self.Di_Y = Di_Y.to(device)
        
        self.optimizer_G_XtoY = optimizer_G_XtoY
        self.optimizer_G_YtoX = optimizer_G_YtoX
        self.optimizer_Di_X = optimizer_Di_X
        self.optimizer_Di_Y = optimizer_Di_Y

        self.scheduler_G_XtoY = scheduler_G_XtoY
        self.scheduler_G_YtoX = scheduler_G_YtoX
        self.scheduler_Di_X = scheduler_Di_X
        self.scheduler_Di_Y = scheduler_Di_Y

        self.use_amp = use_amp and (device == 'cuda')
        self.scaler = GradScaler(enabled=self.use_amp)
        self.device = device

        self.mse_loss = nn.MSELoss()

    def train_step(self, real_x, real_y, epoch):
        self.G_XtoY.train()
        self.G_YtoX.train()
        self.Di_X.train()
        self.Di_Y.train()

        # Step 1: Add input noise that decays over epochs
        noise_std = 0.1 / (epoch + 1)
        real_x_noisy = real_x + torch.randn_like(real_x) * noise_std
        real_y_noisy = real_y + torch.randn_like(real_y) * noise_std

        # --- Train Generators (G_XtoY and G_YtoX) ---
        self.optimizer_G_XtoY.zero_grad()
        self.optimizer_G_YtoX.zero_grad()

        with autocast(enabled=self.use_amp):
            # Generator XtoY forward
            fake_y = self.G_XtoY(real_x_noisy)
            cycled_x = self.G_YtoX(fake_y)
            same_y = self.G_XtoY(real_y_noisy)
            # Discriminator Y returns two scale predictions
            disc_fake_y_s1, disc_fake_y_s2 = self.Di_Y(fake_y)

            # Generator YtoX forward
            fake_x = self.G_YtoX(real_y_noisy)
            cycled_y = self.G_XtoY(fake_x)
            same_x = self.G_YtoX(real_x_noisy)
            # Discriminator X returns two scale predictions
            disc_fake_x_s1, disc_fake_x_s2 = self.Di_X(fake_x)

            # Generator adversarial losses averaged over both scales
            g_loss_XtoY = (self.mse_loss(disc_fake_y_s1, torch.ones_like(disc_fake_y_s1)) + 
                           self.mse_loss(disc_fake_y_s2, torch.ones_like(disc_fake_y_s2))) * 0.5
            
            g_loss_YtoX = (self.mse_loss(disc_fake_x_s1, torch.ones_like(disc_fake_x_s1)) + 
                           self.mse_loss(disc_fake_x_s2, torch.ones_like(disc_fake_x_s2))) * 0.5

            loss_cycle_X = torch.mean((real_x_noisy - cycled_x) ** 2) * 2.5
            loss_cycle_Y = torch.mean((real_y_noisy - cycled_y) ** 2) * 2.5

            loss_id_X = torch.mean((real_x_noisy - same_x) ** 2)
            loss_id_Y = torch.mean((real_y_noisy - same_y) ** 2)

            total_g_loss_XtoY = g_loss_XtoY + loss_cycle_X + loss_id_Y
            total_g_loss_YtoX = g_loss_YtoX + loss_cycle_Y + loss_id_X

        # Backward and Step for Generators
        total_g_loss = total_g_loss_XtoY + total_g_loss_YtoX
        if self.use_amp:
            self.scaler.scale(total_g_loss).backward()
            
            self.scaler.step(self.optimizer_G_XtoY)
            self.scaler.step(self.optimizer_G_YtoX)
        else:
            total_g_loss.backward()
            
            self.optimizer_G_XtoY.step()
            self.optimizer_G_YtoX.step()

        # --- Train Discriminators (Di_X and Di_Y) ---
        self.optimizer_Di_X.zero_grad()
        self.optimizer_Di_Y.zero_grad()

        with autocast(enabled=self.use_amp):
            # Forward Di_X and Di_Y on real & fake
            disc_real_x_s1, disc_real_x_s2 = self.Di_X(real_x_noisy)
            disc_fake_x_s1, disc_fake_x_s2 = self.Di_X(fake_x.detach())

            disc_real_y_s1, disc_real_y_s2 = self.Di_Y(real_y_noisy)
            disc_fake_y_s1, disc_fake_y_s2 = self.Di_Y(fake_y.detach())

            # Discriminator losses for scale 1
            loss_Di_X_s1 = (self.mse_loss(disc_real_x_s1, torch.ones_like(disc_real_x_s1)) + 
                            self.mse_loss(disc_fake_x_s1, torch.zeros_like(disc_fake_x_s1))) * 0.5
            loss_Di_Y_s1 = (self.mse_loss(disc_real_y_s1, torch.ones_like(disc_real_y_s1)) + 
                            self.mse_loss(disc_fake_y_s1, torch.zeros_like(disc_fake_y_s1))) * 0.5

            # Discriminator losses for scale 2
            loss_Di_X_s2 = (self.mse_loss(disc_real_x_s2, torch.ones_like(disc_real_x_s2)) + 
                            self.mse_loss(disc_fake_x_s2, torch.zeros_like(disc_fake_x_s2))) * 0.5
            loss_Di_Y_s2 = (self.mse_loss(disc_real_y_s2, torch.ones_like(disc_real_y_s2)) + 
                            self.mse_loss(disc_fake_y_s2, torch.zeros_like(disc_fake_y_s2))) * 0.5

            # Combined losses
            loss_Di_X = (loss_Di_X_s1 + loss_Di_X_s2) * 0.5
            loss_Di_Y = (loss_Di_Y_s1 + loss_Di_Y_s2) * 0.5
            loss_D = loss_Di_X + loss_Di_Y

        # Backward and Step for Discriminators
        if self.use_amp:
            self.scaler.scale(loss_D).backward()
            
            self.scaler.step(self.optimizer_Di_X)
            self.scaler.step(self.optimizer_Di_Y)
            self.scaler.update()
        else:
            loss_D.backward()
            
            self.optimizer_Di_X.step()
            self.optimizer_Di_Y.step()

        # Step lr schedulers if provided (decays per training step)
        if self.scheduler_G_XtoY is not None:
            self.scheduler_G_XtoY.step()
        if self.scheduler_G_YtoX is not None:
            self.scheduler_G_YtoX.step()
        if self.scheduler_Di_X is not None:
            self.scheduler_Di_X.step()
        if self.scheduler_Di_Y is not None:
            self.scheduler_Di_Y.step()

        return (
            total_g_loss_XtoY.item(),
            total_g_loss_YtoX.item(),
            loss_Di_X.item(),
            loss_Di_Y.item(),
            fake_y.detach(),
            fake_x.detach()
        )

    def test_step(self, real_x, real_y):
        self.G_XtoY.eval()
        self.G_YtoX.eval()
        self.Di_X.eval()
        self.Di_Y.eval()

        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                fake_y = self.G_XtoY(real_x)
                fake_x = self.G_YtoX(real_y)

                cycled_x = self.G_YtoX(fake_y)
                cycled_y = self.G_XtoY(fake_x)

                same_x = self.G_YtoX(real_x)
                same_y = self.G_XtoY(real_y)

                # Forward Di_X and Di_Y on real & fake
                disc_real_x_s1, disc_real_x_s2 = self.Di_X(real_x)
                disc_fake_x_s1, disc_fake_x_s2 = self.Di_X(fake_x)

                disc_real_y_s1, disc_real_y_s2 = self.Di_Y(real_y)
                disc_fake_y_s1, disc_fake_y_s2 = self.Di_Y(fake_y)

                # Generator losses
                g_loss_XtoY = (self.mse_loss(disc_fake_y_s1, torch.ones_like(disc_fake_y_s1)) + 
                               self.mse_loss(disc_fake_y_s2, torch.ones_like(disc_fake_y_s2))) * 0.5
                
                g_loss_YtoX = (self.mse_loss(disc_fake_x_s1, torch.ones_like(disc_fake_x_s1)) + 
                               self.mse_loss(disc_fake_x_s2, torch.ones_like(disc_fake_x_s2))) * 0.5

                loss_cycle_X = torch.mean((real_x - cycled_x) ** 2) * 2.5
                loss_cycle_Y = torch.mean((real_y - cycled_y) ** 2) * 2.5

                loss_id_X = torch.mean((real_x - same_x) ** 2)
                loss_id_Y = torch.mean((real_y - same_y) ** 2)

                total_g_loss_XtoY = g_loss_XtoY + loss_cycle_X + loss_id_Y
                total_g_loss_YtoX = g_loss_YtoX + loss_cycle_Y + loss_id_X

                # Discriminator losses
                loss_Di_X_s1 = (self.mse_loss(disc_real_x_s1, torch.ones_like(disc_real_x_s1)) + 
                                self.mse_loss(disc_fake_x_s1, torch.zeros_like(disc_fake_x_s1))) * 0.5
                loss_Di_Y_s1 = (self.mse_loss(disc_real_y_s1, torch.ones_like(disc_real_y_s1)) + 
                                self.mse_loss(disc_fake_y_s1, torch.zeros_like(disc_fake_y_s1))) * 0.5

                loss_Di_X_s2 = (self.mse_loss(disc_real_x_s2, torch.ones_like(disc_real_x_s2)) + 
                                self.mse_loss(disc_fake_x_s2, torch.zeros_like(disc_fake_x_s2))) * 0.5
                loss_Di_Y_s2 = (self.mse_loss(disc_real_y_s2, torch.ones_like(disc_real_y_s2)) + 
                                self.mse_loss(disc_fake_y_s2, torch.zeros_like(disc_fake_y_s2))) * 0.5

                loss_Di_X = (loss_Di_X_s1 + loss_Di_X_s2) * 0.5
                loss_Di_Y = (loss_Di_Y_s1 + loss_Di_Y_s2) * 0.5

        return (
            total_g_loss_XtoY.item(),
            total_g_loss_YtoX.item(),
            loss_Di_X.item(),
            loss_Di_Y.item()
        )

    def fit(self, train_loader, test_loader, epochs, sample_step_interval=500, checkpoint_interval=10, checkpoint_dir="models", sample_dir="sample", start_epoch=0):
        """Main CycleGAN training orchestration loop"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        
        # We need an iterator for testing samples to save predictions at the end of each epoch
        test_iter = iter(test_loader)
        
        for epoch in range(start_epoch, epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            
            # Reset running averages
            avg_g_XtoY, avg_g_YtoX, avg_d_X, avg_d_Y = 0.0, 0.0, 0.0, 0.0
            n_steps = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for step, (real_x, real_y) in enumerate(loop):
                real_x = real_x.to(self.device)
                real_y = real_y.to(self.device)
                
                g_XtoY_loss, g_YtoX_loss, d_X_loss, d_Y_loss, _, _ = self.train_step(real_x, real_y, epoch)
                
                avg_g_XtoY += g_XtoY_loss
                avg_g_YtoX += g_YtoX_loss
                avg_d_X += d_X_loss
                avg_d_Y += d_Y_loss
                n_steps += 1
                
                # Save predictions and current images every `sample_step_interval` steps
                if (step + 1) % sample_step_interval == 0:
                    save_training_samples(real_x, real_y, self.G_XtoY, self.G_YtoX, epoch, step + 1, output_dir=sample_dir)
                    
                loop.set_postfix({
                    "G_XtoY": f"{g_XtoY_loss:.4f}",
                    "G_YtoX": f"{g_YtoX_loss:.4f}",
                    "D_X": f"{d_X_loss:.4f}",
                    "D_Y": f"{d_Y_loss:.4f}"
                })
            
            # Print epoch average losses
            print(f"Epoch {epoch+1} Train Averages: "
                   f"G_XtoY: {avg_g_XtoY/n_steps:.4f} | "
                   f"G_YtoX: {avg_g_YtoX/n_steps:.4f} | "
                   f"D_X: {avg_d_X/n_steps:.4f} | "
                   f"D_Y: {avg_d_Y/n_steps:.4f}")
            
            # Epoch evaluation on test dataset
            print("Evaluating on test dataset...")
            test_g_XtoY, test_g_YtoX, test_d_X, test_d_Y = 0.0, 0.0, 0.0, 0.0
            test_steps = 0
            
            for real_x_test, real_y_test in test_loader:
                real_x_test = real_x_test.to(self.device)
                real_y_test = real_y_test.to(self.device)
                t_g_XtoY, t_g_YtoX, t_d_X, t_d_Y = self.test_step(real_x_test, real_y_test)
                
                test_g_XtoY += t_g_XtoY
                test_g_YtoX += t_g_YtoX
                test_d_X += t_d_X
                test_d_Y += t_d_Y
                test_steps += 1
                
            print(f"Epoch {epoch+1} Test Averages: "
                  f"G_XtoY: {test_g_XtoY/test_steps:.4f} | "
                  f"G_YtoX: {test_g_YtoX/test_steps:.4f} | "
                  f"D_X: {test_d_X/test_steps:.4f} | "
                  f"D_Y: {test_d_Y/test_steps:.4f}")
            
            # Save test prediction images
            try:
                real_x_sample, real_y_sample = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                real_x_sample, real_y_sample = next(test_iter)
                
            real_x_sample = real_x_sample.to(self.device)
            real_y_sample = real_y_sample.to(self.device)
            save_test_samples(real_x_sample, real_y_sample, self.G_XtoY, self.G_YtoX, epoch, output_dir=os.path.join(sample_dir, "test"))
            
            # Save latest checkpoint after every epoch
            self.save_checkpoint(epoch + 1, checkpoint_dir, is_latest=True)
            
            # Save epoch-specific checkpoint every interval
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1, checkpoint_dir, is_latest=False)
                
    def save_checkpoint(self, epoch, save_dir, is_latest=False):
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'G_XtoY': self.G_XtoY.state_dict(),
            'G_YtoX': self.G_YtoX.state_dict(),
            'Di_X': self.Di_X.state_dict(),
            'Di_Y': self.Di_Y.state_dict(),
            'opt_G_XtoY': self.optimizer_G_XtoY.state_dict(),
            'opt_G_YtoX': self.optimizer_G_YtoX.state_dict(),
            'opt_Di_X': self.optimizer_Di_X.state_dict(),
            'opt_Di_Y': self.optimizer_Di_Y.state_dict()
        }
        filename = 'checkpoint_latest.pt' if is_latest else f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, os.path.join(save_dir, filename))
        if is_latest:
            print(f"Saved latest checkpoint for epoch {epoch} to {save_dir}")
        else:
            print(f"Saved checkpoint for epoch {epoch} to {save_dir}")
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.G_XtoY.load_state_dict(checkpoint['G_XtoY'])
        self.G_YtoX.load_state_dict(checkpoint['G_YtoX'])
        self.Di_X.load_state_dict(checkpoint['Di_X'])
        self.Di_Y.load_state_dict(checkpoint['Di_Y'])
        self.optimizer_G_XtoY.load_state_dict(checkpoint['opt_G_XtoY'])
        self.optimizer_G_YtoX.load_state_dict(checkpoint['opt_G_YtoX'])
        self.optimizer_Di_X.load_state_dict(checkpoint['opt_Di_X'])
        self.optimizer_Di_Y.load_state_dict(checkpoint['opt_Di_Y'])
        print(f"Loaded checkpoint from {checkpoint_path} starting at epoch {checkpoint['epoch']}")
        return checkpoint['epoch']


class PatchNCELoss(nn.Module):
    """
    Patch-wise Noise Contrastive Estimation (PatchNCE) loss.
    Enforces representation similarity between corresponding patches of input and output.
    """
    def __init__(self, num_patches=256, temperature=0.07):
        super().__init__()
        self.num_patches = num_patches
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, feat_query, feat_key):
        # feat_query: [B, N, projection_dim] -> Output features
        # feat_key: [B, N, projection_dim] -> Input features
        batch_size, num_patches, dim = feat_query.shape
        
        # Normalize to unit sphere
        feat_query = nn.functional.normalize(feat_query, dim=-1)
        feat_key = nn.functional.normalize(feat_key, dim=-1)
        
        # Compute cosine similarity matrix: [B, N, N]
        logits = torch.bmm(feat_query, feat_key.permute(0, 2, 1)) / self.temperature
        
        # Target for query patch `i` is the corresponding key patch `i`
        targets = torch.arange(num_patches, dtype=torch.long, device=feat_query.device)
        targets = targets.unsqueeze(0).expand(batch_size, -1) # [B, N]
        
        loss = self.cross_entropy_loss(logits.reshape(-1, num_patches), targets.reshape(-1))
        return loss


class CUTTrainer:
    """
    Trainer class for Contrastive Unpaired Translation (CUT) in PyTorch.
    Optimizes a single-direction translation pipeline with PatchNCE content preservation.
    """
    def __init__(
        self,
        G,
        D,
        F,
        optimizer_G,
        optimizer_D,
        scheduler_G=None,
        scheduler_D=None,
        use_amp=True,
        device='cuda'
    ):
        self.G = G.to(device)
        self.D = D.to(device)
        self.F = F.to(device)
        
        self.opt_G = optimizer_G
        self.opt_D = optimizer_D
        
        self.scheduler_G = scheduler_G
        self.scheduler_D = scheduler_D
        
        self.use_amp = use_amp and (device == 'cuda')
        self.scaler = GradScaler(enabled=self.use_amp)
        self.device = device
        
        self.mse_loss = nn.MSELoss()
        self.nce_loss = PatchNCELoss(num_patches=256, temperature=0.07).to(device)

    def train_step(self, real_x, real_y, epoch):
        self.G.train()
        self.D.train()
        self.F.train()

        # Input noise decay
        noise_std = 0.1 / (epoch + 1)
        real_x_noisy = real_x + torch.randn_like(real_x) * noise_std
        real_y_noisy = real_y + torch.randn_like(real_y) * noise_std

        # --- Train Generator G and Projector F ---
        self.opt_G.zero_grad()

        with autocast(enabled=self.use_amp):
            # Forward G
            fake_y, feat_real_x = self.G.forward_with_features(real_x_noisy)
            
            # Discriminator predictions
            disc_fake_y_s1, disc_fake_y_s2 = self.D(fake_y)
            
            # GAN loss
            loss_G_GAN = (self.mse_loss(disc_fake_y_s1, torch.ones_like(disc_fake_y_s1)) + 
                          self.mse_loss(disc_fake_y_s2, torch.ones_like(disc_fake_y_s2))) * 0.5
            
            # PatchNCE loss: fake_y vs real_x
            feat_fake_y = self.G.forward_with_features(fake_y, encode_only=True)
            feat_real_x_proj, patch_ids = self.F(feat_real_x, num_patches=256)
            feat_fake_y_proj, _ = self.F(feat_fake_y, num_patches=256, patch_ids=patch_ids)
            
            loss_NCE = 0.0
            for f_q, f_k in zip(feat_fake_y_proj, feat_real_x_proj):
                loss_NCE += self.nce_loss(f_q, f_k)
            loss_NCE /= len(feat_fake_y_proj)
            
            # Identity PatchNCE loss: fake_y_identity vs real_y
            fake_y_identity, feat_real_y = self.G.forward_with_features(real_y_noisy)
            feat_fake_y_idt = self.G.forward_with_features(fake_y_identity, encode_only=True)
            
            feat_real_y_proj, patch_ids_idt = self.F(feat_real_y, num_patches=256)
            feat_fake_y_idt_proj, _ = self.F(feat_fake_y_idt, num_patches=256, patch_ids=patch_ids_idt)
            
            loss_NCE_idt = 0.0
            for f_q, f_k in zip(feat_fake_y_idt_proj, feat_real_y_proj):
                loss_NCE_idt += self.nce_loss(f_q, f_k)
            loss_NCE_idt /= len(feat_fake_y_idt_proj)
            
            # Combined generator loss
            total_g_loss = loss_G_GAN + loss_NCE * 1.0 + loss_NCE_idt * 1.0

        if self.use_amp:
            self.scaler.scale(total_g_loss).backward()
            self.scaler.step(self.opt_G)
        else:
            total_g_loss.backward()
            self.opt_G.step()

        # --- Train Discriminator D ---
        self.opt_D.zero_grad()

        with autocast(enabled=self.use_amp):
            disc_real_y_s1, disc_real_y_s2 = self.D(real_y_noisy)
            disc_fake_y_s1, disc_fake_y_s2 = self.D(fake_y.detach())

            loss_D_real = (self.mse_loss(disc_real_y_s1, torch.ones_like(disc_real_y_s1)) + 
                           self.mse_loss(disc_real_y_s2, torch.ones_like(disc_real_y_s2))) * 0.5
            loss_D_fake = (self.mse_loss(disc_fake_y_s1, torch.zeros_like(disc_fake_y_s1)) + 
                           self.mse_loss(disc_fake_y_s2, torch.zeros_like(disc_fake_y_s2))) * 0.5
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5

        if self.use_amp:
            self.scaler.scale(loss_D).backward()
            self.scaler.step(self.opt_D)
            self.scaler.update()
        else:
            loss_D.backward()
            self.opt_D.step()

        # Step lr schedulers if provided
        if self.scheduler_G is not None:
            self.scheduler_G.step()
        if self.scheduler_D is not None:
            self.scheduler_D.step()

        return (
            loss_G_GAN.item(),
            loss_NCE.item(),
            loss_D.item(),
            fake_y.detach(),
            fake_y_identity.detach()
        )

    def test_step(self, real_x, real_y):
        self.G.eval()
        self.D.eval()
        self.F.eval()

        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                fake_y, feat_real_x = self.G.forward_with_features(real_x)
                
                # Discriminator predictions
                disc_fake_y_s1, disc_fake_y_s2 = self.D(fake_y)
                
                loss_G_GAN = (self.mse_loss(disc_fake_y_s1, torch.ones_like(disc_fake_y_s1)) + 
                              self.mse_loss(disc_fake_y_s2, torch.ones_like(disc_fake_y_s2))) * 0.5
                
                feat_fake_y = self.G.forward_with_features(fake_y, encode_only=True)
                feat_real_x_proj, patch_ids = self.F(feat_real_x, num_patches=256)
                feat_fake_y_proj, _ = self.F(feat_fake_y, num_patches=256, patch_ids=patch_ids)
                
                loss_NCE = 0.0
                for f_q, f_k in zip(feat_fake_y_proj, feat_real_x_proj):
                    loss_NCE += self.nce_loss(f_q, f_k)
                loss_NCE /= len(feat_fake_y_proj)
                
                disc_real_y_s1, disc_real_y_s2 = self.D(real_y)
                disc_fake_y_s1, disc_fake_y_s2 = self.D(fake_y)

                loss_D_real = (self.mse_loss(disc_real_y_s1, torch.ones_like(disc_real_y_s1)) + 
                               self.mse_loss(disc_real_y_s2, torch.ones_like(disc_real_y_s2))) * 0.5
                loss_D_fake = (self.mse_loss(disc_fake_y_s1, torch.zeros_like(disc_fake_y_s1)) + 
                               self.mse_loss(disc_fake_y_s2, torch.zeros_like(disc_fake_y_s2))) * 0.5
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5

        return (
            loss_G_GAN.item(),
            loss_NCE.item(),
            loss_D.item()
        )

    def fit(self, train_loader, test_loader, epochs, sample_step_interval=500, checkpoint_interval=10, checkpoint_dir="models", sample_dir="sample", start_epoch=0):
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        
        test_iter = iter(test_loader)
        
        for epoch in range(start_epoch, epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} (CUT Mode) ---")
            
            avg_g_gan, avg_g_nce, avg_d = 0.0, 0.0, 0.0
            n_steps = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for step, (real_x, real_y) in enumerate(loop):
                real_x = real_x.to(self.device)
                real_y = real_y.to(self.device)
                
                g_gan, g_nce, d_loss, fake_y, _ = self.train_step(real_x, real_y, epoch)
                
                avg_g_gan += g_gan
                avg_g_nce += g_nce
                avg_d += d_loss
                n_steps += 1
                
                if (step + 1) % sample_step_interval == 0:
                    # Save intermediate prediction results
                    save_training_samples(real_x, real_y, self.G, self.G, epoch, step + 1, output_dir=sample_dir)
                    
                loop.set_postfix({
                    "G_GAN": f"{g_gan:.4f}",
                    "G_NCE": f"{g_nce:.4f}",
                    "D": f"{d_loss:.4f}"
                })
            
            print(f"Epoch {epoch+1} Train Averages: "
                  f"G_GAN: {avg_g_gan/n_steps:.4f} | "
                  f"G_NCE: {avg_g_nce/n_steps:.4f} | "
                  f"D: {avg_d/n_steps:.4f}")
            
            # Evaluation
            print("Evaluating on test dataset...")
            test_g_gan, test_g_nce, test_d = 0.0, 0.0, 0.0
            test_steps = 0
            
            for real_x_test, real_y_test in test_loader:
                real_x_test = real_x_test.to(self.device)
                real_y_test = real_y_test.to(self.device)
                t_gan, t_nce, t_d = self.test_step(real_x_test, real_y_test)
                
                test_g_gan += t_gan
                test_g_nce += t_nce
                test_d += t_d
                test_steps += 1
                
            print(f"Epoch {epoch+1} Test Averages: "
                  f"G_GAN: {test_g_gan/test_steps:.4f} | "
                  f"G_NCE: {test_g_nce/test_steps:.4f} | "
                  f"D: {test_d/test_steps:.4f}")
            
            try:
                real_x_sample, real_y_sample = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                real_x_sample, real_y_sample = next(test_iter)
                
            real_x_sample = real_x_sample.to(self.device)
            real_y_sample = real_y_sample.to(self.device)
            save_test_samples(real_x_sample, real_y_sample, self.G, self.G, epoch, output_dir=os.path.join(sample_dir, "test"))
            
            # Save checkpoints
            self.save_checkpoint(epoch + 1, checkpoint_dir, is_latest=True)
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1, checkpoint_dir, is_latest=False)
                
    def save_checkpoint(self, epoch, save_dir, is_latest=False):
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'F': self.F.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict()
        }
        filename = 'checkpoint_latest.pt' if is_latest else f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, os.path.join(save_dir, filename))
        if is_latest:
            print(f"Saved latest CUT checkpoint for epoch {epoch} to {save_dir}")
        else:
            print(f"Saved CUT checkpoint for epoch {epoch} to {save_dir}")
            
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Generator load
        if 'G_XtoY' in checkpoint:
            self.G.load_state_dict(checkpoint['G_XtoY'])
        elif 'G' in checkpoint:
            self.G.load_state_dict(checkpoint['G'])
            
        # Discriminator load
        if 'Di_Y' in checkpoint:
            self.D.load_state_dict(checkpoint['Di_Y'])
        elif 'D' in checkpoint:
            self.D.load_state_dict(checkpoint['D'])
            
        # Projector load
        if 'F' in checkpoint:
            self.F.load_state_dict(checkpoint['F'])
            
        # Optimizer load (only load if resuming from CUT checkpoint)
        if 'opt_G' in checkpoint:
            self.opt_G.load_state_dict(checkpoint['opt_G'])
        if 'opt_D' in checkpoint:
            self.opt_D.load_state_dict(checkpoint['opt_D'])
            
        print(f"Loaded checkpoint from {checkpoint_path} (CUT compatible) starting at epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
