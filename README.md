# NVIDIALibraries.jl
NVIDIALibraries.jl was designed to allow full low level access by providing wrapper functions to the CUDA Toolkit libraries.

This package is a Julia library written for Julia 1.0 and onwards.

## What does full low level access mean?
NVIDIALibraries.jl defines C definitions and functions as they would appear in a C project.

## Which NVIDIA libraries are currently supported?
- CUDA library (NVIDIALibraries.CUDA)
- CUDA Runtime library (NVIDIALibraries.CUDARuntime)
- CUBLAS library (NVIDIALibraries.CUBLAS)

## How do I add this Julia package?
Enter the Pkg REPL-mode from the Julia REPL by using the key `]`.

Now add the Julia package by using the `add` command:
```
add https://github.com/mikhail-j/NVIDIALibraries.jl
```

## How are CUDA device arrays managed?
The `CUDAArray` datatype uses the CUDA Runtime library to allocate and free GPU device memory.

CUDAArrays can be constructed with initial values by passing a Julia array to `CUDAArray()`.

In addition, CUDAArrays can be explicitly freed by calling `deallocate!(::CUDAArray)`.

## How do I load a CUDA kernel from a file?
Use `cuModuleLoad()` to load kernel by filename.
```julia
example_module = cuModuleLoad("example.ptx")
```

Alternatively, you can use the CUDA JIT linker to compile a PTX file into a cubin format.
```julia
jit_options = [CU_JIT_TARGET, CU_JIT_OPTIMIZATION_LEVEL]
jit_option_values = [CU_TARGET_COMPUTE_70, Cuint(4)]
jit_link_state = cuLinkCreate(2, jit_options, jit_option_values)
cuLinkAddFile(jit_link_state, CU_JIT_INPUT_PTX, "example.ptx")
cubin_array = cuLinkComplete(jit_link_state)

# cleanup CUDA JIT linker
result = cuLinkDestroy(jit_link_state)
@assert (result == CUDA_SUCCESS) ("cuLinkDestroy() error: " * cuGetErrorString(result))
```

Then, load the kernel from the new cubin image as a CUDA module.
```julia
example_module = cuModuleLoadData(Base.unsafe_convert(Ptr{Nothing}, cubin_array))
```

## What should I do if I recently installed or uninstalled a newer CUDA Toolkit version?
For systems with multiple CUDA Toolkit versions installed, please manually precompile this module in order for it to recognize the new library paths.

Manual precompilation of NVIDIALibraries.jl can be done with the following command:
```julia
Base.compilecache(Base.identify_package("NVIDIALibraries"))
```

## License
The license can be found in `LICENSE`.
