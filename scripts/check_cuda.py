import sys

import torch

print(f"PYTHON_VERSION={sys.version}")
print(f"TORCH_VERSION={torch.__version__}")
print(f"TORCH_CUDA_VERSION={torch.version.cuda}")
print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"DEVICE_COUNT={torch.cuda.device_count()}")
    print(f"DEVICE_NAME={torch.cuda.get_device_name(0)}")
