## Comparison of Sandwich and LipKernel at inference

import torch 
import torch.nn as nn
import numpy as np
import time
from layer import *
import matplotlib.pyplot as plt

def comp_time(c,s,k,stride=1,padding=1,batch_size=1, runs=10):

    # Create input tensor with batch size 1 and channel size c
    u = torch.randn(batch_size, c, s, s)

    total_time_LipKernel = 0.0
    total_time_Sandwich = 0.0
    total_time_Orthogon = 0.0
    
    for _ in range(runs):
        # Reinitialize the layers for each run
        LipKernel_layer = nn.Conv2d(c, c, k, stride, padding)
        Sandwich_layer = SandwichConv(c, c, k, stride, padding, scale=1)
        Orthogon_layer = OrthogonConv(c, c, k, stride, padding, scale=1)
        
        # Measure time for LipKernel layer
        start_time = time.time()
        _ = LipKernel_layer(u)
        end_time = time.time()
        total_time_LipKernel += end_time - start_time
        
        # Measure time for Sandwich layer
        start_time = time.time()
        _ = Sandwich_layer(u)
        end_time = time.time()
        total_time_Sandwich += end_time - start_time

        # Measure time for Sandwich layer
        start_time = time.time()
        _ = Orthogon_layer(u)
        end_time = time.time()
        total_time_Orthogon += end_time - start_time
    
    # Return average times
    return total_time_LipKernel / runs * 1000, total_time_Sandwich / runs * 1000, total_time_Orthogon / runs * 1000

## Runtime vs. # Chanenels n = 8

# Channel sizes to test
channel_sizes = [10,20,50,100,150,200,250,300,350,400,450,500]
s = 32
k = 3

# Lists to store the computation times
times_LipKernel_channels = []
times_Sandwich_channels = []
times_Orthogon_channels = []

for c in channel_sizes:
    print("Channel size: ", c)
    time_LipKernel_channels, time_Sandwich_channels, time_Orthogon_channels = comp_time(c,s,k)
    times_LipKernel_channels.append(time_LipKernel_channels)
    times_Sandwich_channels.append(time_Sandwich_channels)
    times_Orthogon_channels.append(time_Orthogon_channels)

# Plotting the results
#plt.figure(figsize=(10, 6))
#plt.plot(channel_sizes, times_LipKernel_channels*1000, label="LipKernel", marker='o')
#plt.plot(channel_sizes, times_Sandwich_channels*1000, label="Sandwich", marker='s')
#plt.xlabel("Channel Size")
#plt.ylabel("Avg. ms")
#plt.title("Computation Time vs Channel Size for LipKernel and Sandwich")
#plt.legend()
#plt.grid(True)
#plt.show()

np.savetxt("lipkernel_times_channels.csv", np.column_stack([channel_sizes, times_LipKernel_channels]), delimiter=",", header="Channel Size, LipKernel Time")
np.savetxt("sandwich_times_channels.csv", np.column_stack([channel_sizes, times_Sandwich_channels]), delimiter=",", header="Channel Size, Sandwich Time")
np.savetxt("orthogon_times_channels.csv", np.column_stack([channel_sizes, times_Orthogon_channels]), delimiter=",", header="Channel Size, Orthogon Time")

# Runtime vs spatial size cin=cout=32

c = 32
spatial_sizes = [8,16,32,64,128,256,512]
k = 3

# Lists to store the computation times
times_LipKernel_spatial = []
times_Sandwich_spatial = []
times_Orthogon_spatial = []

for s in spatial_sizes:
    print("Image size: ", s)
    time_LipKernel_spatial, time_Sandwich_spatial, time_Orthogon_spatial = comp_time(c,s,k)
    times_LipKernel_spatial.append(time_LipKernel_spatial)
    times_Sandwich_spatial.append(time_Sandwich_spatial)
    times_Orthogon_spatial.append(time_Orthogon_spatial)

# Plotting the results
#plt.figure(figsize=(10, 6))
#plt.plot(spatial_sizes, times_LipKernel_spatial*1000, label="LipKernel", marker='o')
#plt.plot(spatial_sizes, times_Sandwich_spatial*1000, label="Sandwich", marker='s')
#plt.xlabel("Spatial Size")
#plt.ylabel("Avg. ms")
#plt.title("Computation Time vs Spatial Size for LipKernel and Sandwich")
#plt.legend()
#plt.grid(True)
#plt.show()

np.savetxt("lipkernel_times_spatial.csv", np.column_stack([spatial_sizes, times_LipKernel_spatial]), delimiter=",", header="Spatial Size, LipKernel Time")
np.savetxt("sandwich_times_spatial.csv", np.column_stack([spatial_sizes, times_Sandwich_spatial]), delimiter=",", header="Spatial Size, Sandwich Time")
np.savetxt("orthogon_times_spatial.csv", np.column_stack([spatial_sizes, times_Orthogon_spatial]), delimiter=",", header="Spatial Size, Orthogon Time")

# Runtime vs Kernel Size, cin= cout=32, n=16

c = 32
s = 32
kernel_sizes = [3,5,7,9,11,13,15]

# Lists to store the computation times
times_LipKernel_kernel = []
times_Sandwich_kernel = []
times_Orthogon_kernel = []

for k in kernel_sizes:
    print("Kernel size: ", k)
    time_LipKernel_kernel, time_Sandwich_kernel, time_Orthogon_kernel = comp_time(c,s,k)
    times_LipKernel_kernel.append(time_LipKernel_kernel)
    times_Sandwich_kernel.append(time_Sandwich_kernel)
    times_Orthogon_kernel.append(time_Orthogon_kernel)

# Plotting the results
#plt.figure(figsize=(10, 6))
#plt.plot(kernel_sizes, times_LipKernel_kernel*1000, label="LipKernel", marker='o')
#plt.plot(kernel_sizes, times_Sandwich_kernel*1000, label="Sandwich", marker='s')
#plt.xlabel("Kernel Size")
#plt.ylabel("Avg. ms")
#plt.title("Computation Time vs Kernel Size for LipKernel and Sandwich")
#plt.legend()
#plt.grid(True)
#plt.show()

np.savetxt("lipkernel_times_kernel.csv", np.column_stack([kernel_sizes, times_LipKernel_kernel]), delimiter=",", header="Kernel Size, LipKernel Time")
np.savetxt("sandwich_times_kernel.csv", np.column_stack([kernel_sizes, times_Sandwich_kernel]), delimiter=",", header="Kernel Size, Sandwich Time")
np.savetxt("orthogon_times_kernel.csv", np.column_stack([kernel_sizes, times_Orthogon_kernel]), delimiter=",", header="Kernel Size, Orthogon Time")