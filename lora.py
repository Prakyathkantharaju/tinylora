from tinygrad import Tensor
from tinygrad.nn import Linear


class LoraLinear:
  def __init__(self, 
               old_layer, 
               rank: int = 8, 
               alpha: float = 0.9,
               old_grad: bool= False) -> None:
    self.old_layer = old_layer  # old layer.
    layer_shape = old_layer.weight.shape
    self.lora_a = Linear(layer_shape[1], rank, bias=False)
    self.lora_b = Linear(rank, layer_shape[0], bias=False)
    self.scaling = alpha # Need a better way to store this. This is not tinygrad style. 
    self.old_layer.weight.grad = old_grad
    if self.old_layer.bias is not None:
      self.old_layer.bias.grad = old_grad
  
  def __call__(self, x: Tensor) -> Tensor:
    return self.lora_b(self.lora_a(x)) * self.scaling + self.old_layer(x)

class LoraTensor:
  def __init__(self,
               old_layer, 
               rank: int = 8,
               alpha: float = 0.9,
               old_grad: bool= False) -> None:
    self.old_layer = old_layer  # old layer.
    layer_shape = old_layer.shape
    self.lora_a = Tensor.kaiming_uniform(layer_shape[1], rank) # Should I have use this sampling or just use uniform sampling?
    self.lora_b = Tensor.kaiming_uniform(rank, layer_shape[0])
    self.scaling = alpha # Need a better way to store this. This is not tinygrad style. 
    self.old_layer.grad = old_grad
  
  def __call__(self, x: Tensor) -> Tensor:
    return self.lora_b.dot(self.lora_a.dot(x)) * self.scaling + self.old_layer.dot(x)

def MakeLora(old_model,
              config: dict = None) -> None:
    model, config = old_model, config # better way to do this ?
    for name, layer in model.__dict__.items():
      if isinstance(layer, Linear):
        setattr(model, name, LoraLinear(layer, **config))
      elif isinstance(layer, Tensor):
        setattr(model, name, LoraTensor(layer, **config))
      else:
        setattr(model, name, layer)

    old_parameters =  sum([layer.old_layer.weight.numel() for layer in model.__dict__.values() if isinstance(layer, LoraLinear)])
    old_parameters += sum([layer.old_layer.numel() for layer in model.__dict__.values() if isinstance(layer, LoraTensor)])
    new_parameters =  sum([layer.lora_a.weight.numel() + layer.lora_b.weight.numel() for name, layer in model.__dict__.items() if isinstance(layer, LoraLinear)])
    new_parameters += sum([layer.lora_a.numel() + layer.lora_b.numel() for name, layer in model.__dict__.items() if isinstance(layer, LoraTensor)])
    print(f"full parameters: {old_parameters}, New parameters: {new_parameters}")
    return model

