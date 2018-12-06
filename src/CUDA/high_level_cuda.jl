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

