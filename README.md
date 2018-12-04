# NVIDIALibraries.jl
NVIDIALibraries.jl was designed to allow full low level access by providing wrapper functions to the CUDA Toolkit libraries.

This package is a Julia library written for Julia 1.0 and onwards.

## How do I add this Julia package?
Enter the Pkg REPL-mode from the Julia REPL by using the key `]`.

Now add the Julia package by using the `add` command:
```
add https://github.com/mikhail-j/NVIDIALibraries.jl
```

## What should I do if I recently installed or uninstalled a newer CUDA Toolkit version?
For systems with multiple CUDA Toolkit versions installed, please manually precompile this module in order for it to recognize the new library paths.

Manual precompilation of NVIDIALibraries.jl can be done with the following command:
```julia
Base.compilecache(Base.identify_package("NVIDIALibraries"))
```

## License
The license can be found in `LICENSE`.
