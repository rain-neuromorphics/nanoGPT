from torch import Tensor
import torch.nn as nn
from typing import Tuple, Callable, Dict, Optional
import torch
from model import LayerNorm, CausalSelfAttention, MLP, GPT
from torch.nn import functional as F

def make_learnable_by_zo(
    model: GPT,
    linearize_output_loss: bool = True
) -> None:
    
    def apply_forward_hook(mod: nn.Module) -> None:
        if isinstance(mod, LayerNorm) or isinstance(mod, CausalSelfAttention) or isinstance(mod, MLP):
            mod.register_forward_hook(forward_hook)
    
    def apply_backward_hook(mod: nn.Module) -> None:
        if isinstance(mod, LayerNorm) or isinstance(mod, CausalSelfAttention) or isinstance(mod, MLP):
            mod.register_full_backward_hook(backward_hook)
    
    # apply hooks on transformer blocks
    model.transformer.apply(apply_forward_hook)
    model.transformer.apply(apply_backward_hook)
    
    # apply hooks on lm head 
    model.lm_head.register_forward_hook(forward_hook)
    model.lm_head.register_full_backward_hook(
        backward_hook if linearize_output_loss \
        else lambda mod, grad_input, grad_output: backward_hook(mod, grad_input, grad_output, model.loss_fn)
    )
    
    # apply hooks on loss module
    model.loss_fn.register_forward_hook(hacky_hook_forward)
    model.loss_fn.register_full_backward_hook(hacky_hook_backward if linearize_output_loss else hook_backward_output_loss)
    
def forward_hook(
    module: nn.Module,
    input: Tensor,
    output: Tensor
) -> Tuple[Tensor] | None:
    
    module.register_buffer("inputs", input[0])

def hacky_hook_forward(
    module: nn.Module,
    input: Tuple[Tensor],
    output: Tensor  
)-> Tuple[Tensor] | None:
    
    module.register_buffer("targets", input[1])


def hacky_hook_backward(
    module: nn.Module,
    grad_input: Tensor,
    grad_output: Tensor,    
)-> Tuple[Tensor] | None:
    """
    1) one_hot_targets = F.one_hot(module.targets, num_classes=grad_input[0].size(-1)).float()
    2) targets = one_hot_targets.argmax(dim=-1)
    => targets.equal(one_hot_targets)
    """
    
    one_hot_targets = F.one_hot(module.targets, num_classes=grad_input[0].size(-1)).float()
    return (one_hot_targets, grad_input[1])

def hook_backward_output_loss(
    module: nn.Module,
    grad_input: Tensor,
    grad_output: Tensor,    
)-> Tuple[Tensor] | None:
    """
    1) one_hot_targets = F.one_hot(module.targets, num_classes=grad_input[0].size(-1)).float()
    2) targets = one_hot_targets.argmax(dim=-1)
    => targets.equal(one_hot_targets)
    """

    return grad_input

def backward_hook(
    module: nn.Module,
    grad_input: Tensor,
    grad_output: Tensor,
    loss_fn: Optional[Callable] = None
) -> Tuple[Tensor] | None:
    
    def weight_perturbation(module: nn.Module, loss_fn: Callable, eps: float = 1e-6) -> None:
            # create a dictionary of perturbations  
            wps = {n: torch.randn_like(p) for n, p in module.named_parameters()}
            
            # apply positive weight perturbation (w + eps * u) and measure loss
            for n, p in module.named_parameters():
                p.add_(eps * wps[n])
            loss_p = loss_fn(module(module.inputs)) # assume input buffer was built during a forward pass with a forward hook
            
            # apply negative weight perturbation (w - eps * u = w + eps * u - 2 eps * u) and measure loss
            for n, p in module.named_parameters():
                p.add_(-2 * eps * wps[n])
            
            loss_n = loss_fn(module(module.inputs)) # assume input buffer was built during a forward pass with a forward hook    
            
            # reset weights to their initial value by adding eps * u (w - eps * u + eps * u = w)
            for n, p in module.named_parameters():
                p.add_(eps * wps[n])
            
            # manually populate parameter gradients
            for n, p in module.named_parameters():
                p.grad = (1 / (2 * eps)) * (loss_p - loss_n) * wps[n]

    def input_perturbation(module: nn.Module, loss_fn: Callable, eps: float = 1e-6) -> Tensor:
        # create an input perturbation          
        ip = torch.randn_like(module.inputs)     # assume input buffer was built during a forward pass with a forward hook (with clone + detach)
        loss_p = loss_fn(module(module.inputs + eps * ip))
        loss_n = loss_fn(module(module.inputs - 2 * eps * ip))
        return (1 / (2 * eps)) * (loss_p - loss_n) * ip
        
    # zero out module gradients computed by standard 
    module.zero_grad()
    
    # create local loss function
    if loss_fn is None:
        loss_fn_ = lambda x: (x * grad_output[0]).sum()
        # to be checked
    else:
        targets = grad_output[0].argmax(dim=-1)
        loss_fn_ = lambda x: loss_fn.forward(x, targets)
    
    weight_perturbation(module, loss_fn_)
    grad_input = input_perturbation(module, loss_fn_)
    return grad_input,

class CustomCrossEntropy(nn.Module):
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)