import torch
from torch import nn


class ContrastiveClasifier(nn.Module):
    """Contrastive Classifier.

    Calculates the distance between two random vectors, and returns an exponential transformation of it,
    which can be interpreted as the logits for the two vectors being different.

    p : Probability of x1 and x2 being different

    p = 1 - exp( -dist(x1,x2) )
    """

    def __init__(
        self,
        distance: nn.Module,
    ):
        """
        Args:
            distance : A Pytorch module which takes two (batches of) vectors and returns a (batch of)
                positive number.
        """
        super().__init__()

        self.distance = distance

        self.eps = 1e-10

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
        x8: torch.Tensor,
        weight_fft_branch: float,
    ) -> torch.Tensor:

        # Compute distance
        xt = torch.cat((x1, x5), -1)
        yt = torch.cat((x2, x6), -1)
        
        xf = torch.cat((x3, x7), -1)
        yf = torch.cat((x4, x8), -1)
        
        distst = self.distance(xt,yt)
        distsf = self.distance(xf,yf)
        
        dists = torch.maximum(distst, weight_fft_branch*distsf)

        # Probability of the two embeddings being equal: exp(-dist)
        log_prob_equal = -dists

        # Computation of log_prob_different
        prob_different = torch.clamp(1 - torch.exp(log_prob_equal), self.eps, 1)
        log_prob_different = torch.log(prob_different)

        logits_different = log_prob_different - log_prob_equal

        return logits_different
