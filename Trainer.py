import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10


#Save the plots to /plot directory
def show_plot(x, x_label, y, y_label, title):
    import os
    plt.plot(x,y, color="maroon")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    file_path = os.path.join('plots', title+'png')
    plt.savefig(file_path)
    plt.close()


def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        #Save the value of the args
        self.args = args
        self.current_epoch = current_epoch

        #Build the "planning" for beta
        if self.args.kl_anneal_type == 'Cyclical':
            self.beta_schedule = self.cyclical(
                n_iter=args.num_epoch,
                start=0.0,
                stop=1.0,
                n_cycle=args.kl_anneal_cycle,
                ratio=args.kl_anneal_ratio
            )
        elif self.args.kl_anneal_type == "Linear":
            self.beta_schedule = self.linear(n_iter = args.num_epoch)

        elif self.args.kl_anneal_type == 'none':
            self.beta_schedule = self.no_kl_annealing(n_iter = args.num_epoch)

        elif self.args.kl_anneal_type == 'Hybrid_CF':
            self.beta_schedule = self.hybrid_cyclical_flat(
                n_iter=args.num_epoch,
                n_change = args.kl_anneal_change,
                start=0.0,
                stop=1.0,
                n_cycle=args.kl_anneal_cycle,
                ratio=args.kl_anneal_ratio
            )

        elif self.args.kl_anneal_type == 'Hybrid_LF':
            self.beta_schedule = self.hybrid_linear_flat(n_iter=args.num_epoch, n_change = args.kl_anneal_change)

        #Save beta
        self.beta = self.beta_schedule[self.current_epoch]
        
    def update(self):
        #update the epoch
        self.current_epoch += 1
        if len(self.beta_schedule) > self.current_epoch:
            self.beta = self.beta_schedule[self.current_epoch]
        else:
            self.beta = self.beta_schedule[-1]
    
    def get_beta(self):
        return self.beta

    def cyclical(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # Total list to store all beta values
        beta_schedule = []

        # How many steps in one cycle?
        period = n_iter // n_cycle

        # How many steps are used to increase from start to stop?
        step_up = int(period * ratio)

        for cycle in range(n_cycle):
            # For each step in the cycle
            for i in range(period):
                if i < step_up:
                    # Linearly increase from start to stop
                    beta_value = start + (stop - start) * (i / step_up)
                else:
                    # After step_up, just stay at max (stop)
                    beta_value = stop
                beta_schedule.append(beta_value)

        # In case n_iter is not a perfect multiple, cut off extra values
        return beta_schedule[:n_iter]
    
    def linear(self, n_iter):
        beta_schedule = [0]
        slope = 1/n_iter
        for i in range(1,n_iter):
            beta_schedule.append(beta_schedule[-1]+slope)
        return beta_schedule
    
    def no_kl_annealing(self, n_iter):
        return [1 for i in range (n_iter)]
    
    def hybrid_cyclical_flat(self, n_iter, n_change, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        beta_schedule = [] # Total list to store all beta values
        period = n_change // n_cycle # How many steps in one cycle?
        step_up = int(period * ratio) # How many steps are used to increase from start to stop?
        for cycle in range(n_cycle): # For each step in the cycle
            for i in range(period):
                if i < step_up:
                    beta_value = start + (stop - start) * (i / step_up) # Linearly increase from start to stop
                else:
                    beta_value = stop # After step_up, just stay at max (stop)
                beta_schedule.append(beta_value)
        beta_schedule = beta_schedule[:n_change] # In case n_iter is not a perfect multiple, cut off extra values

        while len(beta_schedule)<n_iter:
            beta_schedule.append(1)    #Fill the end with ones

        return beta_schedule

    def hybrid_linear_flat(self, n_iter, n_change):
        beta_schedule = [0]
        slope = 1/n_change
        for i in range(1,n_change):
            beta_schedule.append(beta_schedule[-1]+slope)
        while len(beta_schedule) < n_iter:
            beta_schedule.append(1)
        return beta_schedule

        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label, previous_image, state):
        # Encode the image into feature space using RGB_Encoder
        img_encoded = self.frame_transformation(img)
        previous_image_encoded = self.frame_transformation(previous_image)
        # Encode the label into feature space using Label_Encoder
        label_encoded = self.label_transformation(label)
        
        if state=="training":
            # Use the Gaussian_Predictor to generate 'z', 'mu' and 'logvar' values
            z, mu, logvar = self.Gaussian_Predictor(img_encoded, label_encoded)
        elif state=="testing":
            # Generatez from a normal law
            z = torch.randn(1, args.N_dim, self.args.frame_H, self.args.frame_W).to(self.args.device)
            mu=logvar=0

        #concat = torch.reshape(concat, (self.args.batch_size, self.args.train_vi_len*3, self.args.frame_H, self.args.frame_W))
        decoded = self.Decoder_Fusion(previous_image_encoded, label_encoded, z)
        # Use the Generator to generate a new image from the latent variable z
        generated_img = self.Generator(decoded)
        
        # Step 7: Return the generated image along with 'mu' and 'logvar'
        return generated_img, mu, logvar
    

    def training_stage(self):
        torch.autograd.set_detect_anomaly(True)
        print("num epoch : ", self.args.num_epoch, type(self.args.num_epoch))

        # Define the validation and training loss array
        validation_loss_array = []
        training_loss_array = []
        
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            epoch_loss = []
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                epoch_loss.append(loss.item())
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            training_loss = np.mean(epoch_loss)
            print("Training loss : ", training_loss)
            training_loss_array.append(np.mean(epoch_loss))
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            
            validation_loss = self.eval()
            print("Validation loss : ", validation_loss)
            validation_loss_array.append(validation_loss)
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            print("teacher forcing ratio : ", self.tfr)

        # Save the plots
        x=[i for i in range(self.args.num_epoch)]
        print("validation loss array : ", validation_loss_array)
        print("training loss array : ", training_loss_array)
        show_plot(x, "Number of epoch", validation_loss_array, "Validation loss", "Learning curve (validation)")
        show_plot(x, "Number of epoch", training_loss_array, "Training loss", "Learning curve (training)")
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        return loss.item()
    

    def training_one_step(self, img, label, adapt_TeacherForcing):

        # Initialise the previous image to the label
        previous_image = img[:,0]

        predicted_frames = []

        # Do the forward for the 16 frames of the video
        for i in range (self.args.train_vi_len):

            # Forward pass
            generated_img, mu, logvar = self.forward(img[:,i], label[:,i], previous_image, state="training")
            predicted_frames.append(generated_img.unsqueeze(1))

            # Save the previous image (what we generated if no tf / the true image is tf)
            if adapt_TeacherForcing:
                previous_image = img[:,i]
            else:
                previous_image = generated_img

        predicted = torch.cat(predicted_frames, dim=1)
        # Compute the losses (KL divergence and MSE)
        kld_loss = kl_criterion(mu, logvar, batch_size=img.size(0))  # KL Divergence loss
        recon_loss = self.mse_criterion(predicted, img)  # Mean Squared Error between original and generated
        # Combine the losses
        loss = recon_loss + self.kl_annealing.get_beta() * kld_loss  # KL annealing weight
        
        # Backpropagate the gradients
        self.optim.zero_grad()
        loss.backward()
        
        # Update model parameters (optimizer step)
        self.optimizer_step()
        
        # Return the loss to monitor training
        return loss
    
    @torch.no_grad()
    def val_one_step(self, img, label):

        img = img.permute(1, 0, 2, 3, 4).to(self.args.device) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4).to(self.args.device) # change tensor into (seq, B, C, H, W)
        
        # decoded_frame_list is used to store the predicted frame seq
        # label_list is used to store the label seq
        decoded_frame_list = [img[0]]
        label_list = []
        psnr_per_frame=[]


        # Predicting the image for the 630 frames in the video
        for i in tqdm(range(1,label.shape[0])):
            # run the forward pass
            generated_img, mu, logvar = self.forward(decoded_frame_list[i-1], label[i], decoded_frame_list[i-1], state="testing")
            # complete the frame and label lists
            decoded_frame_list.append(generated_img)
            label_list.append(label[i])
            psnr_per_frame.append(Generate_PSNR(img[i], generated_img).item())
        show_plot([i for i in range(len(psnr_per_frame))], "Frame number", psnr_per_frame, "PNSR score", "PSNR per frame")

        generated_frame = stack(decoded_frame_list)
        
        # Compute the losses (PNSR)
        loss = Generate_PSNR(img, generated_frame)
        
        return loss
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.args.tfr_sde:
            self.tfr = max(0.0, self.tfr - self.tfr_d_step) #Not going below zero
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    # Teacher forcing : si on laisse le model predire la suite de ses propres predictions il sera nul parce que ses predictions sont nulles (au debut)
    # Au debut on le force à faire la frame n+2 à partir de la ground truth à n+1 mais au fur et à mesure on le laisse utiliser la propre prédiction de n+& pour inférer n+2
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    # Kl annealing : à quel point on laisse le modele explorer librement. On instaure une sorte de cycle avec des périodes ou il peut explorer et d'autres ou on est plus stricts
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="Either 'Cyclical', 'Linear', 'none', 'Hybrid_CF' or 'Hybrid_LF'")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    parser.add_argument('--kl_anneal_change',    type=int, default=20,              help="The epoch at wich the k1 annealing will change for the hybrid method")
    

    

    args = parser.parse_args()
    
    main(args)
