import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

# 1. Load your existing TorchScript file
model_path = './inference/swinir_x2.pt'
traced_model = torch.jit.load(model_path)

# 2. Optimize for mobile (fuses layers, reduces size)
optimized_model = optimize_for_mobile(traced_model)

# 3. Save as Lite Interpreter format (.ptl)
output_lite = './inference/swinir_x2.ptl'
optimized_model._save_for_lite_interpreter(output_lite)

print(f"✓ Mobile model saved to: {output_lite}")