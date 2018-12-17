#=*
* High level CUBLAS functions
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

function cublasGetErrorName(error_status::cublasStatus_t)::String
    if (error_status == cublasStatus_t(0))
        return "CUBLAS_STATUS_SUCCESS"
    elseif (error_status == cublasStatus_t(1))
        return "CUBLAS_STATUS_NOT_INITIALIZED"
    elseif (error_status == cublasStatus_t(3))
        return "CUBLAS_STATUS_ALLOC_FAILED"
    elseif (error_status == cublasStatus_t(7))
        return "CUBLAS_STATUS_INVALID_VALUE"
    elseif (error_status == cublasStatus_t(8))
        return "CUBLAS_STATUS_ARCH_MISMATCH"
    elseif (error_status == cublasStatus_t(11))
        return "CUBLAS_STATUS_MAPPING_ERROR"
    elseif (error_status == cublasStatus_t(13))
        return "CUBLAS_STATUS_EXECUTION_FAILED"
    elseif (error_status == cublasStatus_t(14))
        return "CUBLAS_STATUS_INTERNAL_ERROR"
    elseif (error_status == cublasStatus_t(15))
        return "CUBLAS_STATUS_NOT_SUPPORTED"
    elseif (error_status == cublasStatus_t(16))
        return "CUBLAS_STATUS_LICENSE_ERROR"
    else
        error("cublasGetErrorName(): invalid cublasStatus_t value, ", error_status, "!")
    end
end

