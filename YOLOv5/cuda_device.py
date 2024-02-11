import torch

def get_available_cuda_devices():
    num_cuda_devices = torch.cuda.device_count()
    if num_cuda_devices == 0:
        print("No CUDA devices found.")
        return

    print(f"Found {num_cuda_devices} CUDA device(s):")
    for i in range(num_cuda_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"CUDA:{i} - {device_name}")

if __name__ == "__main__":
    get_available_cuda_devices()
