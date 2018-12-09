#=*
* High level CUDA functions
*
* Copyright (C) 2018 Qijia (Michael) Jin
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*=#

function cuGetErrorString(error::CUresult)::String
    local verbose_error::Array{Ptr{UInt8}, 1} = [Ptr{UInt8}(C_NULL)]
    local result::CUresult = cuGetErrorString(error, verbose_error)
    @assert (result == CUDA_SUCCESS)
    return unsafe_string(pop!(verbose_error))
end

function cuGetErrorName(error::CUresult)::String
    local verbose_error::Array{Ptr{UInt8}, 1} = [Ptr{UInt8}(C_NULL)]
    local result::CUresult = cuGetErrorName(error, verbose_error)
    @assert (result == CUDA_SUCCESS) ("cuGetErrorName() error: " * cuGetErrorString(result))
    return unsafe_string(pop!(verbose_error))
end

cuInit(Flags::Integer) = cuInit(Cuint(Flags))

function cuDriverGetVersion()::Cint
    local driver_version::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuDriverGetVersion(driver_version)
    @assert (result == CUDA_SUCCESS) ("cuDriverGetVersion() error: " * cuGetErrorString(result))
    return pop!(driver_version)
end

function cuDeviceGet(ordinal::Cint)::CUdevice
    local cuda_device::Array{CUdevice, 1} = zeros(CUdevice, 1)
    local result::CUresult = cuDeviceGet(Base.unsafe_convert(Ptr{CUdevice}, cuda_device), ordinal)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGet() error: " * cuGetErrorString(result))
    return pop!(cuda_device)
end

cuDeviceGet(ordinal::Integer) = cuDeviceGet(Cint(ordinal))

function cuDeviceGetCount()::Cint
    local device_count::Array{CUdevice, 1} = zeros(Cint, 1)
    local result::CUresult = cuDeviceGetCount(device_count)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGetCount() error: " * cuGetErrorString(result))
    return pop!(device_count)
end

function cuDeviceGetName(dev::CUdevice)::String
    # allow up to 50 characters for returned device name
    local device_name::Array{UInt8, 1} = zeros(UInt8, 40)
    local result::CUresult = cuDeviceGetName(Base.unsafe_convert(Ptr{UInt8}, device_name), Cint(40), dev)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGetName() error: " * cuGetErrorString(result))
    return unsafe_string(Base.unsafe_convert(Ptr{UInt8}, device_name))
end

function cuDeviceTotalMem(dev::CUdevice)::Csize_t
    local bytes::Array{Csize_t, 1} = zeros(Csize_t, 1)
    local result::CUresult = cuDeviceTotalMem(bytes, dev)
    @assert (result == CUDA_SUCCESS) ("cuDeviceTotalMem() error: " * cuGetErrorString(result))
    return pop!(bytes)
end

function cuDeviceGetAttribute(attrib::CUdevice_attribute, dev::CUdevice)::Cint
    local cint_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuDeviceGetAttribute(cint_array, attrib, dev)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGetAttribute() error: " * cuGetErrorString(result))
    return pop!(cint_array)
end

function cuDeviceGetProperties(dev::CUdevice)::CUdevprop
    local prop_array::Array{CUdevprop, 1} = zeros(CUdevprop, 1)
    local result::CUresult = cuDeviceGetProperties(prop_array, dev)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGetProperties() error: " * cuGetErrorString(result))
    return pop!(prop_array)
end

function cuDeviceComputeCapability(dev::CUdevice)::Tuple{Cint, Cint}
    local major_array::Array{Cint, 1} = zeros(Cint, 1)
    local minor_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuDeviceComputeCapability(major_array, minor_array, dev)
    @assert (result == CUDA_SUCCESS) ("cuDeviceComputeCapability() error: " * cuGetErrorString(result))
    return (pop!(major_array), pop!(minor_array))
end

# primary context management functions
function cuDevicePrimaryCtxRetain(dev::CUdevice)::CUcontext
    local ctx_array::Array{CUcontext, 1} = [C_NULL]
    local result::CUresult = cuDevicePrimaryCtxRetain(ctx_array, dev)
    @assert (result == CUDA_SUCCESS) ("cuDevicePrimaryCtxRetain() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

function cuDevicePrimaryCtxGetState(dev::CUdevice)::Tuple{CUctx_flags, Cint}
    local flags_array::Array{CUctx_flags, 1} = zeros(CUctx_flags, 1)
    local active_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuDevicePrimaryCtxGetState(dev, flags_array, active_array)
    @assert (result == CUDA_SUCCESS) ("cuDevicePrimaryCtxGetState() error: " * cuGetErrorString(result))
    return (pop!(flags_array), pop!(active_array))
end

# context management functions
function cuCtxCreate(flags::CUctx_flags, dev::CUdevice)::CUcontext
    local ctx_array::Array{CUcontext, 1} = [C_NULL]
    local result::CUresult = cuCtxCreate(ctx_array, flags, dev)
    @assert (result == CUDA_SUCCESS) ("cuCtxCreate() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

function cuCtxPopCurrent()::CUcontext
    local ctx_array::Array{CUcontext, 1} = [C_NULL]
    local result::CUresult = cuCtxPopCurrent(ctx_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxPopCurrent() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

function cuCtxGetCurrent()::CUcontext
    local ctx_array::Array{CUcontext, 1} = [C_NULL]
    local result::CUresult = cuCtxGetCurrent(ctx_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetCurrent() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

function cuCtxGetDevice()::CUdevice
    local dev_array::Array{CUdevice, 1} = [C_NULL]
    local result::CUresult = cuCtxGetDevice(dev_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetDevice() error: " * cuGetErrorString(result))
    return pop!(dev_array)
end

function cuCtxGetFlags()::CUctx_flags
    local ctx_array::Array{CUctx_flags, 1} = zeros(CUctx_flags, 1)
    local result::CUresult = cuCtxGetFlags(ctx_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetFlags() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

function cuCtxGetLimit(limit::CUlimit)::CUlimit
    local limit_value_array::Array{Csize_t, 1} = zeros(Csize_t, 1)
    local result::CUresult = cuCtxGetLimit(limit_value_array, limit)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetLimit() error: " * cuGetErrorString(result))
    return pop!(limit_value_array)
end

function cuCtxGetCacheConfig()::CUfunc_cache
    local func_cache_array::Array{CUfunc_cache, 1} = zeros(CUfunc_cache, 1)
    local result::CUresult = cuCtxGetCacheConfig(func_cache_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetCacheConfig() error: " * cuGetErrorString(result))
    return pop!(func_cache_array)
end

function cuCtxGetSharedMemConfig()::CUsharedconfig
    local shared_cfg_array::Array{CUsharedconfig, 1} = zeros(CUsharedconfig, 1)
    local result::CUresult = cuCtxGetSharedMemConfig(shared_cfg_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetSharedMemConfig() error: " * cuGetErrorString(result))
    return pop!(shared_cfg_array)
end

function cuCtxGetApiVersion(ctx::CUcontext)::Cuint
    local version_array::Array{Cuint, 1} = zeros(Cuint, 1)
    local result::CUresult = cuCtxGetApiVersion(ctx, version_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetApiVersion() error: " * cuGetErrorString(result))
    return pop!(version_array)
end

function cuCtxGetStreamPriorityRange()::Tuple{Cint, Cint}
    local lp_array::Array{Cint, 1} = zeros(Cint, 1)
    local gp_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuCtxGetStreamPriorityRange(lp_array, gp_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetStreamPriorityRange() error: " * cuGetErrorString(result))
    return (pop!(lp_array), pop!(gp_array))
end

function cuCtxAttach()::CUcontext
    local ctx_array::Array{CUcontext, 1} = [C_NULL]
    # context attach flags argument must be 0
    local result::CUresult = cuCtxAttach(ctx_array, Cuint(0))
    @assert (result == CUDA_SUCCESS) ("cuCtxAttach() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

# module management functions
function cuModuleLoad(fname::Ptr{UInt8})::CUmodule
    local module_array::Array{CUmodule, 1} = [C_NULL]
    local result::CUresult = cuModuleLoad(Base.unsafe_convert(Ptr{CUmodule}, module_array), fname)
    @assert (result == CUDA_SUCCESS) ("cuModuleLoad() error: " * cuGetErrorString(result))
    return pop!(module_array)
end

cuModuleLoad(fname::Array{UInt8, 1}) = cuModuleLoad(Base.unsafe_convert(Ptr{UInt8}, fname))
cuModuleLoad(fname::String) = cuModuleLoad(Base.unsafe_convert(Ptr{UInt8}, map(UInt8, collect(fname))))

function cuModuleLoadData(image::Ptr{Nothing})::CUmodule
    local module_array::Array{CUmodule, 1} = [C_NULL]
    local result::CUresult = cuModuleLoadData(module_array, image)
    @assert (result == CUDA_SUCCESS) ("cuModuleLoadData() error: " * cuGetErrorString(result))
    return pop!(module_array)
end

cuModuleLoadData(image::Ptr{UInt8}) = cuModuleLoadData(Ptr{Nothing}(image))
cuModuleLoadData(image::Array{UInt8, 1}) = cuModuleLoadData(Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt8}, image)))
cuModuleLoadData(image::String) = cuModuleLoadData(map(UInt8, collect(image)))

function cuModuleLoadDataEx(image::Ptr{Nothing}, numOptions::Cuint, options::Array{CUjit_option, 1}, optionValues::Array{Ptr{Cvoid}, 1})::CUmodule
    local module_array::Array{CUmodule, 1} = [C_NULL]
    local result::CUresult = cuModuleLoadDataEx(module_array, image, numOptions, options, optionValues)
    @assert (result == CUDA_SUCCESS) ("cuModuleLoadDataEx() error: " * cuGetErrorString(result))
    return pop!(module_array)
end

function cuModuleLoadDataEx(image::Ptr{Nothing}, numOptions::Cuint, options::Ptr{CUjit_option}, optionValues::Ptr{Ptr{Cvoid}})::CUmodule
    local module_array::Array{CUmodule, 1} = [C_NULL]
    local result::CUresult = cuModuleLoadDataEx(Base.unsafe_convert(Ptr{CUmodule}, module_array), image, numOptions, options, optionValues)
    @assert (result == CUDA_SUCCESS) ("cuModuleLoadDataEx() error: " * cuGetErrorString(result))
    return pop!(module_array)
end

function cuModuleLoadFatBinary(fatCubin::Ptr{Nothing})::CUmodule
    local module_array::Array{CUmodule, 1} = [C_NULL]
    local result::CUresult = cuModuleLoadFatBinary(module_array, fatCubin)
    @assert (result == CUDA_SUCCESS) ("cuModuleLoadFatBinary() error: " * cuGetErrorString(result))
    return pop!(module_array)
end

cuModuleLoadFatBinary(fatCubin::Ptr{UInt8}) = cuModuleLoadFatBinary(Ptr{Nothing}(fatCubin))
cuModuleLoadFatBinary(fatCubin::Array{UInt8, 1}) = cuModuleLoadFatBinary(Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt8}, fatCubin)))
cuModuleLoadFatBinary(fatCubin::String) = cuModuleLoadFatBinary(map(UInt8, collect(fatCubin)))

function cuModuleGetFunction(hmod::CUmodule, name::Ptr{UInt8})::CUfunction
    local function_array::Array{CUfunction, 1} = [C_NULL]
    local result::CUresult = cuModuleGetFunction(Base.unsafe_convert(Ptr{CUfunction}, function_array), hmod, name)
    @assert (result == CUDA_SUCCESS) ("cuModuleGetFunction() error: " * cuGetErrorString(result))
    return pop!(function_array)
end

cuModuleGetFunction(hmod::CUmodule, name::Array{UInt8, 1}) = cuModuleGetFunction(hmod, Base.unsafe_convert(Ptr{UInt8}, name))
cuModuleGetFunction(hmod::CUmodule, name::String) = cuModuleGetFunction(hmod, map(UInt8, collect(name)))

function cuModuleGetGlobal(hmod::CUmodule, name::Ptr{UInt8})::Tuple{CUdeviceptr, Csize_t}
    local dptr_array::Array{CUdeviceptr, 1} = [C_NULL]
    local bytes_array::Array{Csize_t, 1} = zeros(Csize_t, 1)
    local result::CUresult = cuModuleGetGlobal(Base.unsafe_convert(Ptr{CUdeviceptr}, dptr_array), bytes_array, hmod, name)
    @assert (result == CUDA_SUCCESS) ("cuModuleGetGlobal() error: " * cuGetErrorString(result))
    return (pop!(dptr_array), pop!(bytes_array))
end

cuModuleGetGlobal(hmod::CUmodule, name::Array{UInt8, 1}) = cuModuleGetGlobal(hmod, Base.unsafe_convert(Ptr{UInt8}, name))
cuModuleGetGlobal(hmod::CUmodule, name::String) = cuModuleGetGlobal(hmod, map(UInt8, collect(name)))

function cuModuleGetTexRef(hmod::CUmodule, name::Ptr{UInt8})::CUtexref
    local texref_array::Array{CUtexref, 1} = [C_NULL]
    local result::CUresult = cuModuleGetTexRef(Base.unsafe_convert(Ptr{CUtexref}, texref_array), hmod, name)
    @assert (result == CUDA_SUCCESS) ("cuModuleGetTexRef() error: " * cuGetErrorString(result))
    return pop!(texref_array)
end

cuModuleGetTexRef(hmod::CUmodule, name::Array{UInt8, 1}) = cuModuleGetTexRef(hmod, Base.unsafe_convert(Ptr{UInt8}, name))
cuModuleGetTexRef(hmod::CUmodule, name::String) = cuModuleGetTexRef(hmod, map(UInt8, collect(name)))

function cuModuleGetSurfRef(hmod::CUmodule, name::Ptr{UInt8})::CUsurfref
    local surfref_array::Array{CUsurfref, 1} = [C_NULL]
    local result::CUresult = cuModuleGetSurfRef(Base.unsafe_convert(Ptr{CUsurfref}, surfref_array), hmod, name)
    @assert (result == CUDA_SUCCESS) ("cuModuleGetSurfRef() error: " * cuGetErrorString(result))
    return pop!(surfref_array)
end

cuModuleGetSurfRef(hmod::CUmodule, name::Array{UInt8, 1}) = cuModuleGetSurfRef(hmod, Base.unsafe_convert(Ptr{UInt8}, name))
cuModuleGetSurfRef(hmod::CUmodule, name::String) = cuModuleGetSurfRef(hmod, map(UInt8, collect(name)))

# memory management functions
function cuMemAlloc(bytesize::Csize_t)::CUdeviceptr
    local device_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
    local result::CUresult = cuMemAlloc(device_ptr_array, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemAlloc() error: " * cuGetErrorString(result))
    return pop!(device_ptr_array)
end

cuMemAlloc(bytesize::Integer) = cuMemAlloc(Csize_t(bytesize))

function cuMemcpyHtoD(dstDevice::CUdeviceptr, dstOffset::Csize_t, srcHost::Array{T}, srcOffset::Csize_t, bytesize::Csize_t)::Nothing where T
    local result::CUresult = cuMemcpyHtoD(dstDevice + dstOffset, Ptr{Nothing}(Base.unsafe_convert(Ptr{T}, srcHost)) + srcOffset, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemcpyHtoD() error: " * cuGetErrorString(result))
    nothing
end

function cuMemcpyHtoD(dstDevice::CUdeviceptr, dstOffset::Csize_t, srcHost::Ptr, srcOffset::Csize_t, bytesize::Csize_t)::Nothing
    local result::CUresult = cuMemcpyHtoD(dstDevice + dstOffset, srcHost + srcOffset, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemcpyHtoD() error: " * cuGetErrorString(result))
    nothing
end

cuMemcpyHtoD(dstDevice::CUdeviceptr, dstOffset::Integer, srcHost::Array, srcOffset::Integer, bytesize::Integer) = cuMemcpyHtoD(dstDevice, Csize_t(dstOffset), srcHost, Csize_t(srcOffset), Csize_t(bytesize))
cuMemcpyHtoD(dstDevice::CUdeviceptr, dstOffset::Integer, srcHost::Ptr, srcOffset::Integer, bytesize::Integer) = cuMemcpyHtoD(dstDevice, Csize_t(dstOffset), srcHost, Csize_t(srcOffset), Csize_t(bytesize))

function cuMemcpyDtoH(dstHost::Array{T}, dstOffset::Csize_t, srcDevice::CUdeviceptr, srcOffset::Csize_t, bytesize::Csize_t)::Nothing where T
    local result::CUresult = cuMemcpyDtoH(Ptr{Nothing}(Base.unsafe_convert(Ptr{T}, dstHost)) + dstOffset, srcDevice + srcOffset, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemcpyDtoH() error: " * cuGetErrorString(result))
    nothing
end

function cuMemcpyDtoH(dstHost::Ptr, dstOffset::Csize_t, srcDevice::CUdeviceptr, srcOffset::Csize_t, bytesize::Csize_t)::Nothing
    local result::CUresult = cuMemcpyDtoH(dstHost + dstOffset, srcDevice + srcOffset, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemcpyDtoH() error: " * cuGetErrorString(result))
    nothing
end

cuMemcpyDtoH(dstHost::Array, dstOffset::Integer, srcDevice::CUdeviceptr, srcOffset::Integer, bytesize::Integer) = cuMemcpyDtoH(dstHost, Csize_t(dstOffset), srcDevice, Csize_t(srcOffset), Csize_t(bytesize))
cuMemcpyDtoH(dstHost::Ptr, dstOffset::Integer, srcDevice::CUdeviceptr, srcOffset::Integer, bytesize::Integer) = cuMemcpyDtoH(dstHost, Csize_t(dstOffset), srcDevice, Csize_t(srcOffset), Csize_t(bytesize))

function cuMemcpyDtoD(dstDevice::CUdeviceptr, dstOffset::Csize_t, srcDevice::CUdeviceptr, srcOffset::Csize_t, bytesize::Csize_t)::Nothing
    local result::CUresult = cuMemcpyDtoD(dstDevice + dstOffset, srcDevice + srcOffset, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemcpyDtoD() error: " * cuGetErrorString(result))
    nothing
end

cuMemcpyDtoD(dstDevice::CUdeviceptr, dstOffset::Integer, srcDevice::CUdeviceptr, srcOffset::Integer, bytesize::Integer) = cuMemcpyDtoD(dstDevice, Csize_t(dstOffset), srcDevice, Csize_t(srcOffset), Csize_t(bytesize))

