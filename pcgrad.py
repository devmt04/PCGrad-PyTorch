import torch
import torch.nn as nn
import numpy as np
import random

from operator import mul
from functools import reduce

class PCGrad():
  # TODO: PARALLEL PROCESSING
  def __init__(self, gradient_list):
    if not isinstance(gradient_list, list):
      raise TypeError("gradient_list must be a list")
    if not gradient_list:
      raise RuntimeError("gradient_list is empty")

    self.gradient_list = gradient_list
    self.num_tasks = len(gradient_list)
    self.device = torch.device("cpu")
    self.verbose = False

  def resolve_grads(self, verbose = False):
    # TODO : Throw error if gradients provided are not from shared backbone
    self.verbose = verbose
    grad_dims = []
    flat_grads_with_dims = list(map(self.flatten_gradient, self.gradient_list))
    flat_grads, grad_dims = zip(*flat_grads_with_dims)
    flat_grads = torch.stack(flat_grads)
    resolved_grads = self.project_gradients(flat_grads)
    if self.verbose:
      print("[PCGrad] Gradient clipping should be applied after PCGrad resolution.")
    return [self.restore_dims(g, dims) for g, dims in zip(resolved_grads, grad_dims)]
  
  def flatten_gradient(self, task_gradient):
    output = []
    grad_dim = []
    for param_grad in task_gradient:
      grad_dim.append(tuple(param_grad.shape))
      output.append(torch.flatten(param_grad))
    flat_grad = torch.cat(output, dim=0)
    return flat_grad, grad_dim

  def project_gradients(self, flat_grads):
    # PCGrad will be applied to shared parameters, which are common across all task heads  
    # This is how we are able to compare gradients from different tasks  
    # despite the fact that task heads may differ in architecture or parameter count
    for i in range(self.num_tasks):
      for j in range(i+1, self.num_tasks):
          inner_product = torch.dot(flat_grads[i], flat_grads[j])
          if(inner_product < 0):
            if self.verbose:
              print(f"[PCGrad] Conflicting gradients detected between task {i} & {j}!")
            # resolve them
            norm_squared = torch.norm(flat_grads[j])**2
            if norm_squared.item() > 1e-10:
              proj_direction = inner_product / norm_squared
              flat_grads[i] = flat_grads[i] - proj_direction * flat_grads[j]
            else:
              if self.verbose:
                print(f"[PCGrad] Zero or near-zero gradient norm found for task {j}!")
          else:
            # let them as they are
            continue
    return flat_grads

  def restore_dims(self, task_grad, grad_dim):
    chunk_sizes = [reduce(mul, dims, 1) for dims in grad_dim]
    grad_chunk = torch.split(task_grad, split_size_or_sections=chunk_sizes)
    resized_chunks = []
    for index, grad in enumerate(grad_chunk):
      grad = torch.reshape(grad, grad_dim[index])
      resized_chunks.append(grad)
    return resized_chunks
  
  def to(self, device):
    print(f"[PCGrad] Ensure that gradients are not already on {device} before transferring with .to().")
    self.device = device
    self.gradient_list = [
      [g.to(device) for g in task_grads]
      for task_grads in self.gradient_list
    ]
    return self
