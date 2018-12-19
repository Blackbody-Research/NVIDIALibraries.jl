#=*
* Julia CUDA Array definitions and functions.
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

mutable struct CUDAArray
    ptr::Ptr{Nothing}
    size::Tuple{Vararg{Int}}
    freed::Bool
    is_device::Bool # or host memory
    element_type::DataType

    # assume 'ptr' was returned from cuMemAlloc()
    function CUDAArray(ptr::Ptr{Nothing}, size::Tuple{Vararg{Int}}, is_device::Bool, element_type::DataType)
        local ca::CUDAArray=new(ptr, size, false, is_device, element_type)
        finalizer(deallocate!, ca)
        return ca
    end

    function CUDAArray(jl_array::Array{T}) where T
        local device_pointer_array::Array{Ptr{Nothing}, 1} = [C_NULL]
        cudaMalloc(device_pointer_array, sizeof(jl_array))

        # allocate new array in device memory
        local devptr::Ptr{Nothing} = pop!(device_pointer_array)

        # copy data to the array in device memory
        cudaMemcpy(devptr, jl_array, sizeof(jl_array), cudaMemcpyHostToDevice)

        local ca::CUDAArray = new(devptr, size(jl_array), false, true, T)
        finalizer(deallocate!, ca)
        return ca
    end
end

# free pointers not yet deallocated
function deallocate!(ca::CUDAArray)
    if (!ca.freed && ca.is_device)
        cudaFree(ca.ptr)
        ca.freed = true
    end
    nothing
end

# copy 'n' elements (offsets are zero indexed)
function unsafe_copyto!(dst::Array{T}, doffset::Csize_t, src::CUDAArray, soffset::Csize_t, n::Integer)::Array where T
    if (src.is_device)
        cudaMemcpy(dst + doffset, src.ptr + soffset, sizeof(T) * n, cudaMemcpyDeviceToHost)
    else
        ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t),
            Ptr{Nothing}(Base.unsafe_convert(Ptr{T}, dst)) + doffset,
            src.ptr + soffset,
            Csize_t(sizeof(T) * n))
    end
    return dst
end

# copy 'n' elements (offsets are zero indexed)
function unsafe_copyto!(dst::CUDAArray, doffset::Csize_t, src::Array{T}, soffset::Csize_t, n::Integer)::Array where T
    if (dst.is_device)
        cudaMemcpy(dst.ptr + doffset, src + soffset, sizeof(T) * n, cudaMemcpyHostToDevice)
    else
        ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t),
            dst.ptr + doffset,
            Ptr{Nothing}(Base.unsafe_convert(Ptr{T}, src)) + soffset,
            Csize_t(sizeof(T) * n))
    end
    return dst
end

unsafe_copyto!(dst::Array, doffset::Integer, src::CUDAArray, soffset::Integer, n::Integer) = unsafe_copyto!(dst, Csize_t(doffset), src, Csize_t(soffset), n)
unsafe_copyto!(dst::CUDAArray, doffset::Integer, src::Array, soffset::Integer, n::Integer) = unsafe_copyto!(dst, Csize_t(doffset), src, Csize_t(soffset), n)

function unsafe_copyto!(dst::CUDAArray, src::CUDAArray)::CUDAArray
    local src_byte_size::Csize_t = sizeof(src.element_type) * reduce(*, src.size)
    if (src.is_device && dst.is_device)
        cudaMemcpy(dst.ptr, src.ptr, src_byte_size, cudaMemcpyDeviceToDevice)
    elseif (!src.is_device && dst.is_device)
        cudaMemcpy(dst.ptr, src.ptr, src_byte_size, cudaMemcpyHostToDevice)
    elseif (src.is_device && !dst.is_device)
        cudaMemcpy(dst.ptr, src.ptr, src_byte_size, cudaMemcpyDeviceToHost)
    else
        ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t), dst.ptr, src.ptr, src_byte_size)
    end
    return dst
end

function copyto!(dst::Array{T}, src::CUDAArray)::Array where T
    @assert (size(dst) == src.size)
    if (src.is_device)
        cudaMemcpy(dst, src.ptr, sizeof(dst), cudaMemcpyDeviceToHost)
    else
        cudaMemcpy(dst, src.ptr, sizeof(dst), cudaMemcpyDeviceToHost)
    end
    return dst
end

function copyto!(dst::CUDAArray, src::Array{T})::CUDAArray where T
    @assert (dst.size == size(src))
    if (dst.is_device)
        cudaMemcpy(dst.ptr, src, sizeof(src), cudaMemcpyHostToDevice)
    else
        cudaMemcpy(dst.ptr, src, sizeof(src), cudaMemcpyHostToDevice)
    end
    return dst
end

function copyto!(dst::CUDAArray, src::CUDAArray)::CUDAArray
    @assert ((sizeof(dst.element_type) * reduce(*, dst.size))
            >= (sizeof(src.element_type) * reduce(*, src.size)))
    return unsafe_copyto!(dst, src)
end

function _cast_cudaarray_args(x::T) where T
    if (T <: CUDAArray)
        return x.ptr
    else
        return x
    end
end

@inline function cuLaunchKernel(func::CUfunction, grid::dim3, block::dim3, types::Tuple{Vararg{DataType}}, args...; kwargs...)
    return cuLaunchKernel(func, grid, block, Tuple{types...}, map(_cast_cudaarray_args, args)...; kwargs...)
end

@generated function cuLaunchKernel(func::CUfunction, grid::dim3, block::dim3, types::Type, args...; stream::CUstream = CUstream(C_NULL), shmem::Integer = 0, kwargs...)
    local result_expr::Expr = Expr(:block)

    push!(result_expr.args, Base.@_inline_meta)

    args_types = types.parameters[1].parameters
    args_syms = Array{Symbol, 1}(undef, length(args))
    args_ptrs = Array{Symbol, 1}(undef, length(args))

    for i in 1:length(args)
        # assign safely referenced data to corresponding symbol
        args_syms[i] = gensym()
        push!(result_expr.args, :($(args_syms[i]) = Base.cconvert($(args_types[i]), args[$i])))
        
        # generate julia expressions to obtain 
        args_ptrs[i] = gensym()
        push!(result_expr.args, :($(args_ptrs[i]) = Base.unsafe_convert($(args_types[i]), $(args_syms[i]))))
    end
    
    append!(result_expr.args, (quote
        GC.@preserve $(args_syms...) begin
            arguments::Array{Any, 1} = [$(args_ptrs...)]
            result::CUresult = cuLaunchKernel(func,
                                grid.x, grid.y, grid.z,
                                block.x, block.y, block.z,
                                Cuint(shmem),
                                stream,
                                Base.unsafe_convert(Ptr{Ptr{Nothing}}, arguments),
                                Ptr{Ptr{Nothing}}(C_NULL))
            @assert (result == CUDA_SUCCESS) ("cuLaunchKernel() error: " * cuGetErrorName(result))
        end
    end).args)

    return result_expr
end

@inline function cudaLaunchKernel(func::Ptr{Nothing}, grid::dim3, block::dim3, types::Tuple{Vararg{DataType}}, args...; kwargs...)
    return cudaLaunchKernel(func, grid, block, Tuple{types...}, map(_cast_cudaarray_args, args)...; kwargs...)
end

@generated function cudaLaunchKernel(func::Ptr{Nothing}, grid::dim3, block::dim3, types::Type, args...; stream::cudaStream_t = cudaStream_t(C_NULL), shmem::Integer = 0, kwargs...)
    local result_expr::Expr = Expr(:block)

    push!(result_expr.args, Base.@_inline_meta)

    args_types = types.parameters[1].parameters
    args_syms = Array{Symbol, 1}(undef, length(args))
    args_ptrs = Array{Symbol, 1}(undef, length(args))

    for i in 1:length(args)
        # assign safely referenced data to corresponding symbol
        args_syms[i] = gensym()
        push!(result_expr.args, :($(args_syms[i]) = Base.cconvert($(args_types[i]), args[$i])))
        
        # generate julia expressions to obtain 
        args_ptrs[i] = gensym()
        push!(result_expr.args, :($(args_ptrs[i]) = Base.unsafe_convert($(args_types[i]), $(args_syms[i]))))
    end
    
    append!(result_expr.args, (quote
        GC.@preserve $(args_syms...) begin
            arguments::Array{Any, 1} = [$(args_ptrs...)]
            result::cudaError_t = cudaLaunchKernel(func,
                                grid,
                                block,
                                Base.unsafe_convert(Ptr{Ptr{Nothing}}, arguments),
                                Csize_t(shmem),
                                stream)
            @assert (result == cudaSuccess) ("cudaLaunchKernel() error: " * unsafe_string(cudaGetErrorName(result)))
        end
    end).args)

    return result_expr
end

function cublasHgemm(handle::cublasHandle_t, ta::Char, tb::Char, alpha::Float16, A::CUDAArray, B::CUDAArray, beta::Float16, C::CUDAArray)::Nothing
    @assert ((A.element_type == Float16) &&
            (B.element_type == Float16) &&
            (C.element_type == Float16))
    local transA::cublasOperation_t
    local transB::cublasOperation_t
    local m::Cint = Cint(C.size[1])
    local n::Cint = Cint(C.size[2])
    local k::Cint = Cint(A.size[2])
    local lda::Cint, ldb::Cint
    local ldc::Cint = m

    if (ta == 'N')
        transA = CUBLAS_OP_N
        lda = m
    elseif (ta == 'T')
        transA = CUBLAS_OP_T
        lda = k
    elseif (ta == 'C')
        transA = CUBLAS_OP_C
        lda = k
    end
    if (tb == 'N')
        transB = CUBLAS_OP_N
        ldb = k
    elseif (tb == 'T')
        transB = CUBLAS_OP_T
        ldb = n
    elseif (tb == 'C')
        transB = CUBLAS_OP_C
        ldb = n
    end
    local result::cublasStatus_t = cublasHgemm(handle, transA, transB, m, n, k, alpha, Ptr{Float16}(A.ptr), lda, Ptr{Float16}(B.ptr), ldb, beta, Ptr{Float16}(C.ptr), ldc)
    @assert (result == cudaSuccess) ("cublasHgemm() error: " * cublasGetErrorName(result))
end

# use cublasSgemm_v2() over legacy cublasSgemm()
function cublasSgemm(handle::cublasHandle_t, ta::Char, tb::Char, alpha::Float32, A::CUDAArray, B::CUDAArray, beta::Float32, C::CUDAArray)::Nothing
    @assert ((A.element_type == Float32) &&
            (B.element_type == Float32) &&
            (C.element_type == Float32))
    local transA::cublasOperation_t
    local transB::cublasOperation_t
    local m::Cint = Cint(C.size[1])
    local n::Cint = Cint(C.size[2])
    local k::Cint = Cint(A.size[2])
    local lda::Cint, ldb::Cint
    local ldc::Cint = m

    if (ta == 'N')
        transA = CUBLAS_OP_N
        lda = m
    elseif (ta == 'T')
        transA = CUBLAS_OP_T
        lda = k
    elseif (ta == 'C')
        transA = CUBLAS_OP_C
        lda = k
    end
    if (tb == 'N')
        transB = CUBLAS_OP_N
        ldb = k
    elseif (tb == 'T')
        transB = CUBLAS_OP_T
        ldb = n
    elseif (tb == 'C')
        transB = CUBLAS_OP_C
        ldb = n
    end
    local result::cublasStatus_t = cublasSgemm_v2(handle, transA, transB, m, n, k, alpha, Ptr{Float32}(A.ptr), lda, Ptr{Float32}(B.ptr), ldb, beta, Ptr{Float32}(C.ptr), ldc)
    @assert (result == cudaSuccess) ("cublasSgemm() error: " * cublasGetErrorName(result))
end

# use cublasDgemm_v2() over legacy cublasDgemm()
function cublasDgemm(handle::cublasHandle_t, ta::Char, tb::Char, alpha::Float64, A::CUDAArray, B::CUDAArray, beta::Float64, C::CUDAArray)::Nothing
    @assert ((A.element_type == Float64) &&
            (B.element_type == Float64) &&
            (C.element_type == Float64))
    local transA::cublasOperation_t
    local transB::cublasOperation_t
    local m::Cint = Cint(C.size[1])
    local n::Cint = Cint(C.size[2])
    local k::Cint = Cint(A.size[2])
    local lda::Cint, ldb::Cint
    local ldc::Cint = m

    if (ta == 'N')
        transA = CUBLAS_OP_N
        lda = m
    elseif (ta == 'T')
        transA = CUBLAS_OP_T
        lda = k
    elseif (ta == 'C')
        transA = CUBLAS_OP_C
        lda = k
    end
    if (tb == 'N')
        transB = CUBLAS_OP_N
        ldb = k
    elseif (tb == 'T')
        transB = CUBLAS_OP_T
        ldb = n
    elseif (tb == 'C')
        transB = CUBLAS_OP_C
        ldb = n
    end
    local result::cublasStatus_t = cublasDgemm_v2(handle, transA, transB, m, n, k, alpha, Ptr{Float64}(A.ptr), lda, Ptr{Float64}(B.ptr), ldb, beta, Ptr{Float64}(C.ptr), ldc)
    @assert (result == cudaSuccess) ("cublasDgemm() error: " * cublasGetErrorName(result))
end

function get_cuda_datatype(dt::DataType)::cudaDataType
    if (dt == __half)
        return CUDA_R_16F
    elseif (dt == __half2)
        return CUDA_C_16F
    elseif (dt == Float32)
        return CUDA_R_32F
    elseif (dt == cuFloatComplex)
        return CUDA_C_32F
    elseif (dt == Float64)
        return CUDA_R_64F
    elseif (dt == cuDoubleComplex)
        return CUDA_C_64F
    elseif (dt == Int8)
        return CUDA_R_8I
    elseif (dt == char2)
        return CUDA_C_8I
    elseif (dt == UInt8)
        return CUDA_R_8U
    elseif (dt == uchar2)
        return CUDA_C_8U
    elseif (dt == Int32)
        return CUDA_R_32I
    elseif (dt == int2)
        return CUDA_C_32I
    elseif (dt == UInt32)
        return CUDA_R_32U
    elseif (dt == uint2)
        return CUDA_C_32U
    else
        error("get_cuda_datatype() error: could not find corresponding cudaDataType for ", dt, "!")
    end
end

function cublasGemmEx(handle::cublasHandle_t, algo::cublasGemmAlgo_t, ta::Char, tb::Char, alpha::T, A::CUDAArray, B::CUDAArray, beta::T, C::CUDAArray)::Nothing where T
    local transA::cublasOperation_t
    local transB::cublasOperation_t
    local m::Cint = Cint(C.size[1])
    local n::Cint = Cint(C.size[2])
    local k::Cint = Cint(A.size[2])
    local lda::Cint, ldb::Cint
    local ldc::Cint = m

    if (ta == 'N')
        transA = CUBLAS_OP_N
        lda = m
    elseif (ta == 'T')
        transA = CUBLAS_OP_T
        lda = k
    elseif (ta == 'C')
        transA = CUBLAS_OP_C
        lda = k
    end
    if (tb == 'N')
        transB = CUBLAS_OP_N
        ldb = k
    elseif (tb == 'T')
        transB = CUBLAS_OP_T
        ldb = n
    elseif (tb == 'C')
        transB = CUBLAS_OP_C
        ldb = n
    end

    @assert (A.element_type == B.element_type)

    local AB_cdt::cudaDataType = get_cuda_datatype(A.element_type)
    local C_cdt::cudaDataType = get_cuda_datatype(C.element_type)
    local compute_cdt::cudaDataType = get_cuda_datatype(T)

    @assert ((AB_cdt, C_cdt, compute_cdt) in ((CUDA_R_16F, CUDA_R_16F, CUDA_R_16F),
                                            (CUDA_R_8I, CUDA_R_32I, CUDA_R_32I),
                                            (CUDA_R_16F, CUDA_R_16F, CUDA_R_32F),
                                            (CUDA_R_8I, CUDA_R_32F, CUDA_R_32F),
                                            (CUDA_R_16F, CUDA_R_32F, CUDA_R_32F),
                                            (CUDA_R_32F, CUDA_R_32F, CUDA_R_32F),
                                            (CUDA_R_64F, CUDA_R_64F, CUDA_R_64F),
                                            (CUDA_C_8I, CUDA_C_32F, CUDA_C_32F),
                                            (CUDA_C_32F, CUDA_C_32F, CUDA_C_32F),
                                            (CUDA_C_64F, CUDA_C_64F, CUDA_C_64F))) ("cublasGemmEx() error: Atype/Btype ("
                                                                                    * string(AB_cdt) * "), Ctype ("
                                                                                    * string(C_cdt) * "), and computeType ("
                                                                                    * string(compute_cdt)
                                                                                    * ") is not a supported combination of cudaDataTypes!")


    local result::cublasStatus_t = cublasGemmEx(handle, transA, transB, m, n, k, alpha, A.ptr, AB_cdt, lda, B.ptr, AB_cdt, ldb, beta, C.ptr, C_cdt, ldc, compute_cdt, algo)
    @assert (result == cudaSuccess) ("cublasGemmEx() error: " * cublasGetErrorName(result))
end
