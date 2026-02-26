import torch
import tensorly as tl
tl.set_backend('pytorch')

def mirror_node_omega(hidden_states, rank=39):
    """
    Apply TT-SVD to hidden state trajectory.
    """
    # Stack last k states
    T = torch.stack(hidden_states, dim=0) 
    
    # Adaptive decomposition
    factors = tl.decomposition.tensor_train(T, rank=rank)
    
    # Return compressed cores (ERPS footprint)
    return factors