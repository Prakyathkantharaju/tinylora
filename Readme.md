# Tiny Lora

Writing lora using the tinygrad library. This is work in progress.
This is similar to Peft library from huggingface link [here](https://github.com/huggingface/peft) but with tinygrad and tinygrad principles.



## Usage

```python
# import your models  as model
from lora import MakeLora
config  = {'rank': 8, 'alpha': 0.9}
model = MakeLora(model, config)
```


# TODO
- [x] Lora for Linear eg in `testing_lora.py`
- [x] Lora for Tensors eg in `testing_lora.py`
- [ ] Lora for Transformers 
- [ ] Lora for CNNs
