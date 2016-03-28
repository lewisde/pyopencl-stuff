# example provided by Roger Pau Monn'e
# Updated for Python3 - lewisde

import pyopencl as cl
import numpy
import sys
# import numpy.linalg as la
# import datetime
from time import time
num_range = int(sys.argv[1])

a = numpy.random.rand(num_range).astype(numpy.float32)
b = numpy.random.rand(num_range).astype(numpy.float32)
c_result = numpy.empty_like(a)

kernel1 = "__kernel void sum(__global const float *a,__global const float *b, \
__global float *c){float div = 2;int loop;int gid = \
get_global_id(0);for(loop=0; loop<"
kernel2 = ";loop++){c[gid] = a[gid] + \
b[gid];c[gid] = c[gid]* (a[gid] + b[gid]);c[gid] = c[gid] * \
(a[gid] / div);}}"

kernel = kernel1 + str(num_range) + kernel2


# Speed in normal CPU usage
time1 = time()
for i in range(num_range):
        for j in range(num_range):
                c_result[i] = a[i] + b[i]
                c_result[i] *= c_result[i]
                c_result[i] = c_result[i] * (a[i] / 2.0)
time2 = time()
print("\033[92m\nExecution time of test: {:0.8f} seconds\033[0m\n\n".format( time2 - time1))

for platform in cl.get_platforms():
    for device in platform.get_devices():
        print("=============================================================")
        print("Platform name:", platform.name)
        print("Platform profile:", platform.profile)
        print("Platform vendor:", platform.vendor)
        print("Platform version:", platform.version)
        print("-------------------------------------------------------------")
        print("Device name: \033[91m{}\033[90m".format(device.name))
        print("Device type:", cl.device_type.to_string(device.type))
        print("Device memory: ", device.global_mem_size // 1024 // 1024, 'MB')
        print("Device max clock speed:", device.max_clock_frequency, 'MHz')
        print("Device compute units:", device.max_compute_units)

        # Simnple speed test
        ctx = cl.Context([device])
        queue = cl.CommandQueue(
            ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

        prg = cl.Program(ctx, kernel).build()

        exec_evt = prg.sum(queue, a.shape, None, a_buf, b_buf, dest_buf)
        exec_evt.wait()
        elapsed = 1e-9 * (exec_evt.profile.end - exec_evt.profile.start)

        print("\033[92m\nExecution time of test: {:0.8f} seconds\033[0m\n\n".format(elapsed))

        c = numpy.empty_like(a)
        cl.enqueue_read_buffer(queue, dest_buf, c).wait()
        error = 0
        for i in range(num_range):
                if c[i] != c_result[i]:
                        error = 1
        if error:
                print("\033[91mResults doesn't match!!\033[0m")
        else:
                print("\033[92mResults OK\033[0m")
