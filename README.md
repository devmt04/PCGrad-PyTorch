# PCGrad-PyTorch
PyTorch implementation for "Gradient Surgery for Multi-Task Learning" https://arxiv.org/abs/2001.06782

For the Tensorflow implementation by the Paper authors, please see https://github.com/tianheyu927/PCGrad

Usage: 

```
"""
grad_list is a list of lists
structured as :
[
task1 gradients: [], 
task2 gradients: [], 
...
taskn gradients:[]
]
"""

grad_list = PCGrad(grad_list).to(device).resolve_grads()
```

Base Code forked from Base code forked from [OrthoDex/PCGrad-PyTorch](https://github.com/OrthoDex/PCGrad-PyTorch)

Changes made:

- Change code architecture
- Minor bug fixes

TODOs:

- Implement parallel processing
