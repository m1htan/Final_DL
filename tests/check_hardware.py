import psutil
import platform
import subprocess
import torch
import GPUtil

# CPU
print("CPU Count:", psutil.cpu_count(logical=True))
print("CPU Physical Cores:", psutil.cpu_count(logical=False))
print("CPU Frequency:", psutil.cpu_freq().current, "MHz")

# RAM
ram = psutil.virtual_memory()
print("Total RAM:", round(ram.total / (1024**3), 2), "GB")
print("Available RAM:", round(ram.available / (1024**3), 2), "GB")

# Disk
disk = psutil.disk_usage('/')
print("Total Disk:", round(disk.total / (1024**3), 2), "GB")
print("Used Disk:", round(disk.used / (1024**3), 2), "GB")

# OS Info
print("OS:", platform.system(), platform.release())

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory Total:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), "GB")


result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)
print(result.stdout)


gpus = GPUtil.getGPUs()
for gpu in gpus:
    print("ID:", gpu.id)
    print("Name:", gpu.name)
    print("Load:", f"{gpu.load*100:.1f}%")
    print("Free Memory:", f"{gpu.memoryFree}MB")
    print("Total Memory:", f"{gpu.memoryTotal}MB")
    print("Temperature:", f"{gpu.temperature}°C")