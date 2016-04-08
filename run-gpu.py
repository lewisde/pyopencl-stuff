# datarun.py
# David Lewis
#
# This will be my first complete OpenCL program and will demonstrate the
# power by iterating from 10^2 to 10^6 visible patterns can be seen when
# graphed. The initial version will output csv for graphing in a spreadsheet.

import pyopencl as cl
import numpy
import sys
from time import time, gmtime, strftime


def pfunc(r):
    time1 = time()
    for i in range(r):
        for j in range(r):
                c_result[i] = a[i] + b[i]
                c_result[i] *= c_result[i]
                c_result[i] = c_result[i] * (a[i] / 2.0)
    time2 = time()
 #   print(time2 - time1)
    return time2 - time1


def clfunc(device, kern):
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
 #   print(elapsed)
    return elapsed

if len(sys.argv) == 4:
    step = int(sys.argv[3])
else:
    step = 1
        

start_time = time()
start_str = strftime("%a, %d %b %Y %H:%M:%S", gmtime())
print('Start time: {}, run from {} to {} by {}'.format(start_str, sys.argv[1], sys.argv[2], step))
    
for i in range(int(sys.argv[1]), int(sys.argv[2]), step):
#    print(i)
    run = []
    a = numpy.random.rand(i).astype(numpy.float32)
    print(a)
    b = numpy.random.rand(i).astype(numpy.float32)
    c_result = numpy.empty_like(a)
    kernel1 = "__kernel void sum(__global const float *a,__global const float *b, __global float *c){int loop;int gid = get_global_id(0);for(loop=0; loop<"
    kernel2 = ";loop++){for(int i=0;i<"
    kernel3 = ";i++){c[gid] = a[gid] + b[gid];c[gid] = c[gid]* (a[gid] + b[gid]);c[gid] = c[gid] * a[gid] / 2;}}}"
    kernel = kernel1 + str(i) + kernel2 + str(i) + kernel3

#    run.append(pfunc(i))

    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if cl.device_type.to_string(device.type) == 'GPU':
#            print("Device type:", cl.device_type.to_string(device.type))
                run.append(clfunc(device, kernel))

    with open('run-{}-{}-{}.out'.format(sys.argv[1], sys.argv[2], step), 'a+') as writer:
        string = '{}, {}\n'.format(i, run[0])
        writer.write(string)
        print(string)
        
end_time = time()
end_str = strftime("%a, %d %b %Y %H:%M:%S", gmtime())
print('End time: {}, run from {} to {} by {}'.format(end_str, sys.argv[1], sys.argv[2], step))  
print('Elapsed time in seconds:', end_time - start_time)
