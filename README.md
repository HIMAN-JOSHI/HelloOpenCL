# HelloOpenCL
This program adds two matrices using OpenCL.

The purpose of this program is to learn and understand the basic steps involved in writing an OpenCL program.

Following are the general steps / template usually observed in an OpenCL program.

1. Get platform's Id (GPU, CPU, Accelerator, etc.)
2. Get device Id for that device on above platform on which you (programmer) want to do OpenCL.
3. Create the context (a context is a state maintaining struct.)
4. Create an OpenCL command queue to put OpenCL commands in it.
5. Write an OpenCL kernel "in a string".
6. Create an OpenCL program for the above kernel.
7. Build the program which has the above kernel (there are APIs for this).
8. If there are any errors in the kernel code then get the build log.
9. If successful, create the OpenCL kernel.
10. Create buffers for the device inputs.
11. Set the OpenCL kernel parameters.
12. Send/Write the data from host to device via command queue.
13. Run/Configre the kernel via command queue.
14. Finish the command queue to allow OpenCL to run all the commands (until now) in the command queue.
15. Get/Read the output data from the device into the host output buffer via command queue.
    
    
