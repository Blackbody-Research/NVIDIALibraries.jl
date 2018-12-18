#=*
* High level CUDA runtime functions
*
* Copyright (C) 2018 Blackbody Research LLC
*       Author: Qijia (Michael) Jin
*
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

function cudaStreamCreate()::cudaStream_t
    local stream_array::Array{cudaStream_t, 1} = [C_NULL]
    local result::cudaError_t = cudaStreamCreate(stream_array)
    @assert (result == cudaSuccess) ("cudaStreamCreate() error: " * unsafe_string(cudaGetErrorString(result)))
    return pop!(stream_array)
end

function cudaStreamCreateWithFlags(flags::Cuint)::cudaStream_t
    local stream_array::Array{cudaStream_t, 1} = [C_NULL]
    local result::cudaError_t = cudaStreamCreateWithFlags(stream_array, flags)
    @assert (result == cudaSuccess) ("cudaStreamCreateWithFlags() error: " * unsafe_string(cudaGetErrorString(result)))
    return pop!(stream_array)
end

cudaMalloc(devPtr::Array{Ptr{Cvoid}, 1}, size::Integer) = cudaMalloc(Base.unsafe_convert(Ptr{Ptr{Nothing}}, devPtr), Csize_t(size))
cudaMalloc(devPtr::Ptr{Ptr{Cvoid}}, size::Integer) = cudaMalloc(devPtr, Csize_t(size))

cudaMemcpy(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, count::Integer, kind::Cuint) = cudaMemcpy(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, Csize_t(count), kind)
cudaMemcpy(dst::Ptr{Cvoid}, src::Array{T}, count::Integer, kind::Cuint) where {T} = cudaMemcpy(dst, Ptr{Cvoid}(Base.unsafe_convert(Ptr{T}, src)), Csize_t(count), kind)
cudaMemcpy(dst::Array{T}, src::Ptr{Cvoid}, count::Integer, kind::Cuint) where {T} = cudaMemcpy(Ptr{Cvoid}(Base.unsafe_convert(Ptr{T}, dst)), src, Csize_t(count), kind)

