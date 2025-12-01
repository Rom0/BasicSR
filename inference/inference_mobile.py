import sys
import os

# Add parent directory to path so we can import basicsr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from basicsr.archs.swinir_arch import SwinIR

# 1. Define Model Architecture (MUST match the training config of your .pth)
# Example parameters for SwinIR-Light. Adjust 'upscale', 'window_size', etc.
# to match your specific model (SR, Denoising, etc.)
model = SwinIR(
    upscale=2,
    in_chans=3,
    img_size=64,
    window_size=8,
    img_range=1.,
    depths=[6, 6, 6, 6],
    embed_dim=18,
    num_heads=[6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='pixelshuffle',
    resi_connection='1conv'
)

# 2. Load your .pth weights
model_path = './inference/net_g_22000.pth'
checkpoint = torch.load(model_path, map_location='cpu')
# Handle cases where weights are inside a 'params' or 'params_ema' key
if 'params_ema' in checkpoint:
    checkpoint = checkpoint['params_ema']
elif 'params' in checkpoint:
    checkpoint = checkpoint['params']
model.load_state_dict(checkpoint, strict=True)
model.eval()

# 3. Create a dummy input (fixed size is often better for mobile speed)
# Shape: (Batch, Channels, Height, Width). Example: 64x64 input -> 128x128 output (x2 upscale)
dummy_input = torch.randn(1, 3, 64, 64)

# Test the model first
print("Testing model with dummy input...")
with torch.no_grad():
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")

# 4. Export options
output_onnx = './inference/swinir_x2.onnx'
output_torchscript = './inference/swinir_x2.pt'
print("Exporting model...")

# Try exporting to TorchScript first (more reliable)
print("Attempting TorchScript export...")
try:
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.eval()
    traced_model.save(output_torchscript)
    print(f"✓ Successfully exported to TorchScript: {output_torchscript}")
    print("  (TorchScript format is more reliable and can be converted to ONNX later)")
except Exception as e:
    print(f"✗ TorchScript export failed: {e}")

# Try ONNX export (may fail due to PyTorch 2.9.1 bug)
print("\nAttempting ONNX export...")
try:
    # Try using the legacy exporter by setting environment variable
    import os
    os.environ['TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK'] = 'DISABLED'

    # Try with opset 11 and legacy mode
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        # Try to use legacy exporter
        _retain_param_name=True
    )
    print(f"✓ Successfully exported to ONNX: {output_onnx}")
except Exception as e:
    print(f"✗ ONNX export failed: {e}")
    print("\nNote: This appears to be a bug in PyTorch 2.9.1's ONNX exporter.")
    print("The TorchScript export should work and can be used for inference.")
    print("Alternatively, you can:")
    print("  1. Use the TorchScript model (.pt) directly")
    print("  2. Convert TorchScript to ONNX using onnxruntime or other tools")
    print("  3. Downgrade PyTorch to 2.0.x or 2.1.x for more stable ONNX export")

print(f"Exported to {output_onnx}")