// This program adds two matrices using OpenCL.

// header files

// standard headers
#include<stdio.h>
#include<stdlib.h> // for exit()

// OpenCL headers
#include<CL/opencl.h> 

// global variables
const int iNumberOfArrayElements = 5;

// type of platform
cl_platform_id oclPlatformID;

// type of device
cl_device_id oclDeviceID;

// cl_context - state maintaining struct
cl_context oclContext;

cl_command_queue oclCommandQueue;

cl_program oclProgram;

cl_kernel oclKernel;

float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutput = NULL;

// OpenCL memory object (cl_mem internally void *)
cl_mem deviceInput1 = NULL;
cl_mem deviceInput2 = NULL;
cl_mem deviceOutput = NULL;

// OpenCL Kernel

// __global -> run and call on device (like __device__ on CUDA)
const char* oclSourceCode =
"__kernel void vecAddGPU(__global float *in1, __global float *in2, __global float *out, int len)"\
"{" \
"int i=get_global_id(0);"\
"if(i < len)"\
"{"\
"out[i]=in1[i]+in2[i];"\
"}"\
"}";

// entry-point function
int main(void) {

    // function declarations
    void cleanup(void);

    // variable declarations
    int size = iNumberOfArrayElements * sizeof(float);

    cl_int result = CL_SUCCESS;

    // code

    // host memory allocation
    hostInput1 = (float*)malloc(size);
    if (hostInput1 == NULL) {
        printf("Host memory allocation failed for hostInput1 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float*)malloc(size);
    if (hostInput2 == NULL) {
        printf("Host memory allocation failed for hostInput2 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutput = (float*)malloc(size);
    if (hostOutput == NULL) {
        printf("Host memory allocation failed for hostOutput array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // filling values into host arrays
    hostInput1[0] = 101.0;
    hostInput1[1] = 102.0;
    hostInput1[2] = 103.0;
    hostInput1[3] = 104.0;
    hostInput1[4] = 105.0;

    hostInput2[0] = 201.0;
    hostInput2[1] = 202.0;
    hostInput2[2] = 203.0;
    hostInput2[3] = 204.0;
    hostInput2[4] = 205.0;

    // 1. Get platform's ID
    /**
     * cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_unit *num_platforms)
     *
     * num_entries   - The number of cl_platform_id entries that can be added to platforms.
     * platforms     - Returns a list of OpenCL platforms found. The cl_platform_id values
     *                 returned in platforms can be used to identify a specific OpenCL platform (like CPU, GPU, Accelerator, etc.)
     * num_platforms - Returns the number of OpenCL platforms available.
     *
     * Returns CL_SUCCESS if the function is executed successfully, else it returns CL_INVALID_VALUE.
     *
    */
    result = clGetPlatformIDs(1, // 1 - as here we are interested in 1 ID only.
        &oclPlatformID, // &oclPlatformID - give the platformID in this variable.
        NULL); // actual number of platforms found is not important here so this parameter value is NULL.
    if (result != CL_SUCCESS) {
        printf("clGetPlatformIDs() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 2. Get OpenCL supporting CPU device's ID
    /* Obtain the list of devices available on a platform
     * cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices)

       platform    - Refers to the platform ID returned by clGetPlatformIDs or can be NULL.
       device_type - A bitfield that identifies the type of OpenCL device. It can be used to query specific OpenCL device or all OpenCL devices available. (like - CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, GL_DEVICE_TYPE_ACCELERATOR, etc.)
       num_entries - The number of device IDs that can be added to devices.
       devices     - A list of OpenCL devices found.
       num_devices - The number of OpenCL devices available that match device_type.
    */
    result = clGetDeviceIDs(oclPlatformID,
        CL_DEVICE_TYPE_GPU,
        1, // here we are interested in only 1 device ID so this param value is 1.
        &oclDeviceID, // variable for storing the deviceID
        NULL); // the number of OpenCL devices available that match device_type is not important here so this value is NULL.

    // 3. Create OpenCL compute context
    /**
     * cl_context clCreateContext(cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void *pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data),  void *user_data, cl_int *errcode_ret)
     *
     * properties  - Specifies a list of context property names and their corresponding values.
     * num_devices - The number of devices specified in the devices argument.
     * devices     - A pointer to a list of unique devices returned by clGetDeviceIDs for a platform.
     * pfn_notify  - A callback function which will be used by the OpenCL implementation to report information on errors.
     *      errinfo            - A pointer to the error string.
     *      private_info & cb  - Represent a pointer to a binary data returned by OpenCL that can be used to log additional information about the error.
     *      user_data          - A pointer to user supplied data.
     * user_data     - Passed as user_data when pfn_notify is called.
     * errorcode_ret - Returns an appropriate error code.
     *
     * Returns CL_SUCCESS if the context is created successfully else it returns NULL.
    */
    oclContext = clCreateContext(NULL, // NULL as the context property param is not needed  here.
        1,
        &oclDeviceID,
        NULL, // NULL as no callback function specified here.
        NULL, // NULL para to the callback function since no callback function here.
        &result);
    if (result != CL_SUCCESS) {

        printf("clCreateContext() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 4. Create command queue
    /*
      cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret)

      context    - A valid OpenCL context.
      device     - Device associated with context.
      properties - Specifies a list of properties for the command-queue. This is a bit-field (like - CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE).
      errcode_ret- Returns appropriate error code. If this is NULL, no error code is returned.

      Returns CL_SUCCESS if the command-queue is created successfully else it returns a NULL value.
    */
    //oclCommandQueue = clCreateCommandQueue(oclContext, oclDeviceID, 0, &result);
    oclCommandQueue = clCreateCommandQueueWithProperties(oclContext, oclDeviceID, 0, &result);
    if (result != CL_SUCCESS) {
        printf("clCreateCommandQueue() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 6. Create OpenCL program from .cl
    /*
        Creates a program object for a context, and loads the source code specified by the text
        strings in the strings array into the program object.

        cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret)

        context     - A valid OpenCL context
        strings     - An array of count pointers to optionally null-terminated character strings that make up the source code.
        lengths     - An array with the number of chars in each string (the string length).
        errcode_ret - Returns an appropriate error code.

        Returns a valid non-zero program object and errcode_ret is set to CL_SUCCESS if the program object is created successfully.
        Otherwise, it returns a NULL value with a valid error code.
    */
    oclProgram = clCreateProgramWithSource(oclContext,
        1, // number of string
        (const char**)&oclSourceCode, // Address of kernel source code (variable used)
        NULL, // length of string (NULL indicates full string).
        &result); // CL_SUCCESS or NULL will be stored in this variable.


    // 7. Build OpenCL program
    /*
        Builds (compiles and links) a program executable from the program source or binary

        cl_int clBuildProgram (cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void(CL_CALLBACK *pfn_notify) (cl_program progran, void *user_data), void *user_data)

        program     - The program object

        device_list - A pointer to the list of devices associated with program.

        num_devices - The number of devices listed in device_list

        options     - A pointer to a null-terminated string of characters that describes the build options to be
                      used for building the program executable.
        pfn_notify  - A callback function for notification called when the program executable has been built (successfully or unsuccessfully).

        user_data   - Passed as an argument when pfn_notify is called.

        Returns CL_SUCCESS if the function executed successfully else it returns error.
    */
    result = clBuildProgram(oclProgram, // program to build.
        0,
        NULL, // array of devices for which this program needs to be build.
        NULL, // options for building
        NULL, // Name of the callback function
        NULL); // Param for the callback functions.
    if (result != CL_SUCCESS) {

        size_t len;
        char buffer[2048];
        result = clGetProgramBuildInfo(oclProgram, oclDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("Program Build Log : %s\n", buffer);
        printf("clBuildProgram() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 8. Create OpenCL kernel by function name that we used in .cl file
    /*
        cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret)

        program     - A program object with successfully built executable.
        kernel_name - A function name in the program declared with the kernel qualifier.
        errcode_ret - Returns an appropriate error code.

    */
    oclKernel = clCreateKernel(oclProgram, // program object
        "vecAddGPU", // kernel name
        &result); // errcode_ret
    if (result != CL_SUCCESS) {
        printf("clCreateKernel() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }


    // 9. device memory allocation
    /*
        cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret)

        context     - A valid OpenCL context used to create the buffer object.
        flags       - A bit-field that is used to specify allocation and usage information such as the memory that should be
                      used to allocate the buffer object and how it will be used.
        size        - The size in bytes of the buffer memory object to be allocated.
        host_ptr    - A pointer to the buffer data that may already be allocated by the application.
        errcode_ret - Returns an appropriate error code.

        Returns a valid non-zero buffer object and errcode_ret is set to CL_SUCCESS if the buffer
        object is created successfully. Otherwise, it returns a NULL value with error.

    */

    deviceInput1 = clCreateBuffer(oclContext,
        CL_MEM_READ_ONLY, // This buffer will be used as read-only.
        size, // how much memory to give ?
        NULL, // address of the existing buffer (if any). NULL in this case.
        &result);
    if (result != CL_SUCCESS) {
        printf("clCreateBuffer() failed for 1st Input Array : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceInput2 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
    if (result != CL_SUCCESS) {
        printf("clCreateBuffer() failed for 2nd Input Array : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceOutput = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, size, NULL, &result);
    if (result != CL_SUCCESS) {
        printf("clCreateBuffer() failed for Output Array : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 10. set 0 based 0th argument i.e. deviceInput1
    /*
        Set the argument value for a specific argument of a kernel.
        cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value)

        kernel    - A valid kernel object.
        arg_index - The argument index. Arguments to the kernel are referred by indices that go from 0 for
                    the leftmost argument to n-1, where n is the total number of arguments declared by a kernel.
        arg_size  - Specifies the size of the argument value.
        arg_value - A pointer to data that should be used as the argumen value for argument specified by arg_index.

    */
    result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void*)&deviceInput1);
    if (result != CL_SUCCESS) {
        printf("clSetKernelArg() failed for 1st argument : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 1st argument i.e. deviceInput2
    result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void*)&deviceInput2);
    if (result != CL_SUCCESS) {
        printf("clSetKernelArg() failed for 2nd argument : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 2nd argument i.e. deviceOutput
    result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void*)&deviceOutput);
    if (result != CL_SUCCESS) {
        printf("clSetKernelArg() failed for 3rd argument : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 3rd argument i.e. len
    result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void*)&iNumberOfArrayElements);
    if (result != CL_SUCCESS) {
        printf("clSetKernelArg() failed for 4th argument : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 11. Write above 'input' device buffer to device memory
    /*
        Enqueue commands to write to a buffer object from host memory.

        cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)

        command_queue  - Is a valid host command-queue in which the write command will be queued.
                         command_queue and buffer must be created with the same OpenCL context.
        buffer         - Refers to a valid buffer object.
        blocking_write - Indicates if the write operations are blocking or nonblocking.
        offset         - The offset in bytes in the buffer object to write to.
        size           - The size in bytes of data being written.
        ptr            - The pointer to buffer in host memory where data is to be written from.
        event_wait_list and
        num _events_in_wait_list - Specify the events that need to complete before this particular command can be executed.
        event          - Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete.
    */

    // similar to cudaMemcpy()
    result = clEnqueueWriteBuffer(oclCommandQueue, // command_queue
        deviceInput1, // buffer to write data
        CL_FALSE, // should wait or add commands parallelly to the command_queue while writing the data.
        0, // start writing data from 0th byte offset
        size,
        hostInput1,
        0,
        NULL,
        NULL);

    if (result != CL_SUCCESS) {
        printf("clEnqueuWriteBuffer() failed for 1st input device buffer : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = clEnqueueWriteBuffer(oclCommandQueue, deviceInput2, CL_FALSE, 0, size, hostInput2, 0, NULL, NULL);
    if (result != CL_SUCCESS) {
        printf("clEnqueuWriteBuffer() failed for 2nd input device buffer : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 12. Kernel configuration
    size_t global_size = 5; // 1-D 5 element array operation
    /*
        Enqueues a command to execute a kernel on a device.

        cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)

        command_queue           - A valid command-queue. The kernel will be queued for execution on the device associated with command_queue.
        kernel                  - A valid kernel object. The OpenCL context associated with kernel and command_queue must be the same.
        work_dim                - The number of dimensions used to specify the global work-items and work-items in the work-group.
        global_work_offset      - Must currently be a NULL value.
        global_work_size        - Points to an array of work_dim unsigned values that describe the number of global work-items in work_dim dimensions that will execute the kernel function.
        local_work_size         - Points to an array of work_dim unsigned values that describe the number of work-items that make up a work-group (also refferd to as the size of the work-group)
        event_wait_list and
        num_events_in_wait_list - Specify events that need to complete before this particular command can be executed. IF event_wait_list is NULL, then this particular command does not wait on any event to complete.
        event                   - Returns an event object that identifies this particular kernel execution instance. Event objects are unique and can be used to identify a particular kernel execution instance later on.

        Returns CL_SUCCESS if the kernel execution was successfully queued. Otherwise, it returns error.
    */
    // "ND" stands for "N" Dimensions.
    result = clEnqueueNDRangeKernel(oclCommandQueue, // command_queue
        oclKernel, // kernel
        1, // dimension of the kernel
        NULL, // Reserved parameter
        &global_size, //global_work_size
        NULL,
        0,
        NULL,
        NULL);
    if (result != CL_SUCCESS) {
        printf("clEnqueueNDRangeKernel() failed : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 13. Finish the command queue to allow OpenCL to run all the commands (until this point) in the command queue.
    /*
        Blocks until all previously queued OpenCL commands in a command-queue are issued to the associated device
        and have completed.

        cl_int clFinish(cl_command_queue command_queue)

        Returns CL_SUCCESS if the function call was executed successfully. It returns CL_INVALID_COMMAND_QUEUE if command
        queue is not a valid command-queue and returns CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
        required by the OpenCL implementation on the host.
    */
    // clFinish is synchronous i.e. it waits for the current command to finish and then moves to the next one.
    clFinish(oclCommandQueue);

    // 14. Read back result from the device (i.e. from deviceOutput) into host output buffer i.e. hostOutput variable.
    /**
     * Enqueue commands to read from a buffer object to host memory.
     *
     * cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t cb, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
     *
     * command_queue - Refers to the command-queue in which the read command will be queued.
     *                 command_queue and buffer must be created with the same OpenCL context.
     * buffer        - Refers to a valid buffer object.
     * blocking_read - Indicates if the read operations are blocking or non-blocking.
     * offset        - The offset in bytes in the buffer object to read from.
     * cb            - The size in bytes of data being read.
     * ptr           - The pointer to buffer in host memory where data is to be read into.
     * event_wait_list,
     * num_events_in_wait_list - Specify events that need to complete before this particular command can be executed.
     * event         - Returns an event object that identifies this particular read command and can be used to query or
     *                 queue a wait for this particular command to complete.
     *
     * Returns CL_SUCCESS if the function is executed successfully else it returns error.
    */
    result = clEnqueueReadBuffer(oclCommandQueue,  // command_queue
        deviceOutput, // buffer to read data from.
        CL_TRUE, // block the read.
        0, // start reading from 0th offset.
        size, // size of data to be read.
        hostOutput, // buffer where data is to be read into.
        0, // no events in wait list so 0.
        NULL,
        NULL);
    if (result != CL_SUCCESS) {
        printf("clEnqueueReadBuffer() failed : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 15. Display the results
    int i;
    for (i = 0; i < iNumberOfArrayElements; i++) {
        printf("%f + %f = %f", hostInput1[i], hostInput2[i], hostOutput[i]);
    }

    // cleanup
    cleanup();

    return(0);
}

void cleanup(void) {

     //code
    if (deviceOutput) {
        /**
         * Decrements the memory object reference count.
         *
         * cl_int clReleaseMemObject(cl_mem memobj)
        */
        clReleaseMemObject(deviceOutput);
        deviceOutput = NULL;
    }

    if (deviceInput2) {
        clReleaseMemObject(deviceInput2);
        deviceInput2 = NULL;
    }

    if (deviceInput1) {
        clReleaseMemObject(deviceInput1);
        deviceInput1 = NULL;
    }

    if (oclKernel) {
        clReleaseMemObject(oclKernel);
        oclKernel = NULL;
    }

    if (oclProgram) {
        clReleaseMemObject(oclProgram);
        oclProgram = NULL;
    }

    if (oclCommandQueue) {
        clReleaseCommandQueue(oclCommandQueue);
        oclCommandQueue = NULL;
    }

    if (oclContext) {
        clReleaseContext(oclContext);
        oclContext = NULL;
    }

    if (hostOutput) {
        free(hostOutput);
        hostOutput = NULL;
    }

    if (hostInput2) {
        free(hostInput2);
        hostInput2 = NULL;
    }

    if (hostInput1) {
        free(hostInput1);
        hostInput1 = NULL;
    }

}