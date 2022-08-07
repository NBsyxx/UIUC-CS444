import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.nn import MSELoss
device = torch.device("cuda:0" if torch.cuda.is_available() else "gpu")


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    label_real = torch.full((logits_real.shape[0],), 1, dtype=torch.float, device=device)
    label_fake = torch.full((logits_fake.shape[0],), 0, dtype=torch.float, device=device)
    loss_real = bce_loss(logits_real,label_real,reduction="mean")
    loss_fake = bce_loss(logits_fake,label_fake,reduction="mean")
#     print(loss_real)
#     print(loss_fake)
    loss = (loss_real * logits_real.shape[0] + loss_fake * logits_fake.shape[0])/(logits_real.shape[0] + logits_fake.shape[0])
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    # label fake is one for generators
    label_fake = torch.full((logits_fake.shape[0],1,2,2), 1, dtype=torch.float, device=device)
    loss = bce_loss(logits_fake,label_fake,reduction="mean")
    ##########       END      ##########
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss = 0.5 * (torch.mean(torch.square(scores_real - 1.0))+ torch.mean(torch.square(scores_fake)))
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss = 0.5 * torch.mean(torch.square(scores_fake - 1.0))
    
    ##########       END      ##########
    
    return loss
