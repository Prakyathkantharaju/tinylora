from lora import MakeLora
from models import LinearNet, TinyNet


model_1 = LinearNet()
model_2 = TinyNet()

config = {
  "rank": 8,
  "alpha": 0.9
}


model = MakeLora(model_1, config)
model = MakeLora(model_2, config)

