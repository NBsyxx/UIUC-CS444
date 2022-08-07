import torch
from torch import nn
# from spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3,spectral_norm = True):
        super(Discriminator, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        def SpectralNormLayer(layer):
            return SpectralNorm(module=layer) if spectral_norm else layer
        
        self.seq = nn.Sequential( 
                        SpectralNormLayer(nn.Conv2d(3, 128, 4, stride=2, padding=1, bias=False)),
                        # nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2),
                        SpectralNormLayer(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2),
                        SpectralNormLayer(nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False)),
                        nn.BatchNorm2d(512),
                        nn.LeakyReLU(0.2),
                        SpectralNormLayer(nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=False)),
                        nn.BatchNorm2d(1024),
                        nn.LeakyReLU(0.2),
                        SpectralNormLayer(nn.Conv2d(1024, 1, 4, stride=2, padding=1, bias=False)),
                        )
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        
        ##########       END      ##########
        x = self.seq(x)
        return x



class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.seq = nn.Sequential( 
                        nn.ConvTranspose2d(noise_dim, 1024, 4, stride=1, padding=0, bias=False),
                        nn.ReLU(),
                        nn.BatchNorm2d(1024),
                        nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False),
                        nn.ReLU(),
                        nn.BatchNorm2d(512),
                        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                        nn.ReLU(),
                        nn.BatchNorm2d(256),
                        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=False),
                        
                        torch.nn.Tanh()
                        )
        
#         self.bn1 = nn.BatchNorm2d(1024)
#         self.bn2 = nn.BatchNorm2d(512)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.bn3 = nn.BatchNorm2d(64)
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.seq(x)
        ##########       END      ##########
#         print(x.shape)
        return x
    
