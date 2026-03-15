import torch

print("=" * 40)
print("PyTorch version:      ", torch.__version__)
print("CUDA available:       ", torch.cuda.is_available())
print("CUDA version:         ", torch.version.cuda)
print("GPU count:            ", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"    Memory total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

if not torch.cuda.is_available():
    print("\n[!] No CUDA GPU detected. Model will run on CPU (very slow).")
    print("    Possible reasons:")
    print("    1. No NVIDIA GPU in this machine")
    print("    2. NVIDIA drivers not installed")
    print("    3. CUDA toolkit not installed")
    print("    4. CUDA_VISIBLE_DEVICES set to a non-existent GPU index")
    print("    5. Running in WSL without GPU passthrough configured")
print("=" * 40)
