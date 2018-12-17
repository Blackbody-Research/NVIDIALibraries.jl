#=*
* Export CUDA runtime library v9.1 definitions and functions
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

export
    # constants
    cudaStream_t,
    cudaEvent_t,
    cudaGraphicsResource_t,
    cudaUUID_t,
    cudaHostAllocDefault,
    cudaHostAllocPortable,
    cudaHostAllocMapped,
    cudaHostAllocWriteCombined,
    cudaHostRegisterDefault,
    cudaHostRegisterPortable,
    cudaHostRegisterMapped,
    cudaHostRegisterIoMemory,
    cudaPeerAccessDefault,
    cudaStreamDefault,
    cudaStreamNonBlocking,
    cudaStreamLegacy,
    cudaStreamPerThread,
    cudaEventDefault,
    cudaEventBlockingSync,
    cudaEventDisableTiming,
    cudaEventInterprocess,
    cudaDeviceScheduleAuto,
    cudaDeviceScheduleSpin,
    cudaDeviceScheduleYield,
    cudaDeviceScheduleBlockingSync,
    cudaDeviceBlockingSync,
    cudaDeviceScheduleMask,
    cudaDeviceMapHost,
    cudaDeviceLmemResizeToMax,
    cudaDeviceMask,
    cudaArrayDefault,
    cudaArrayLayered,
    cudaArraySurfaceLoadStore,
    cudaArrayCubemap,
    cudaArrayTextureGather,
    cudaIpcMemLazyEnablePeerAccess,
    cudaMemAttachGlobal,
    cudaMemAttachHost,
    cudaMemAttachSingle,
    cudaOccupancyDefault,
    cudaOccupancyDisableCachingOverride,
    cudaCpuDeviceId,
    cudaInvalidDeviceId,
    cudaCooperativeLaunchMultiDeviceNoPreSync,
    cudaCooperativeLaunchMultiDeviceNoPostSync,
    cudaError_t,
    cudaSuccess,
    cudaErrorMissingConfiguration,
    cudaErrorMemoryAllocation,
    cudaErrorInitializationError,
    cudaErrorLaunchFailure,
    cudaErrorPriorLaunchFailure,
    cudaErrorLaunchTimeout,
    cudaErrorLaunchOutOfResources,
    cudaErrorInvalidDeviceFunction,
    cudaErrorInvalidConfiguration,
    cudaErrorInvalidDevice,
    cudaErrorInvalidValue,
    cudaErrorInvalidPitchValue,
    cudaErrorInvalidSymbol,
    cudaErrorMapBufferObjectFailed,
    cudaErrorUnmapBufferObjectFailed,
    cudaErrorInvalidHostPointer,
    cudaErrorInvalidDevicePointer,
    cudaErrorInvalidTexture,
    cudaErrorInvalidTextureBinding,
    cudaErrorInvalidChannelDescriptor,
    cudaErrorInvalidMemcpyDirection,
    cudaErrorAddressOfConstant,
    cudaErrorTextureFetchFailed,
    cudaErrorTextureNotBound,
    cudaErrorSynchronizationError,
    cudaErrorInvalidFilterSetting,
    cudaErrorInvalidNormSetting,
    cudaErrorMixedDeviceExecution,
    cudaErrorCudartUnloading,
    cudaErrorUnknown,
    cudaErrorNotYetImplemented,
    cudaErrorMemoryValueTooLarge,
    cudaErrorInvalidResourceHandle,
    cudaErrorNotReady,
    cudaErrorInsufficientDriver,
    cudaErrorSetOnActiveProcess,
    cudaErrorInvalidSurface,
    cudaErrorNoDevice,
    cudaErrorECCUncorrectable,
    cudaErrorSharedObjectSymbolNotFound,
    cudaErrorSharedObjectInitFailed,
    cudaErrorUnsupportedLimit,
    cudaErrorDuplicateVariableName,
    cudaErrorDuplicateTextureName,
    cudaErrorDuplicateSurfaceName,
    cudaErrorDevicesUnavailable,
    cudaErrorInvalidKernelImage,
    cudaErrorNoKernelImageForDevice,
    cudaErrorIncompatibleDriverContext,
    cudaErrorPeerAccessAlreadyEnabled,
    cudaErrorPeerAccessNotEnabled,
    cudaErrorDeviceAlreadyInUse,
    cudaErrorProfilerDisabled,
    cudaErrorProfilerNotInitialized,
    cudaErrorProfilerAlreadyStarted,
    cudaErrorProfilerAlreadyStopped,
    cudaErrorAssert,
    cudaErrorTooManyPeers,
    cudaErrorHostMemoryAlreadyRegistered,
    cudaErrorHostMemoryNotRegistered,
    cudaErrorOperatingSystem,
    cudaErrorPeerAccessUnsupported,
    cudaErrorLaunchMaxDepthExceeded,
    cudaErrorLaunchFileScopedTex,
    cudaErrorLaunchFileScopedSurf,
    cudaErrorSyncDepthExceeded,
    cudaErrorLaunchPendingCountExceeded,
    cudaErrorNotPermitted,
    cudaErrorNotSupported,
    cudaErrorHardwareStackError,
    cudaErrorIllegalInstruction,
    cudaErrorMisalignedAddress,
    cudaErrorInvalidAddressSpace,
    cudaErrorInvalidPc,
    cudaErrorIllegalAddress,
    cudaErrorInvalidPtx,
    cudaErrorInvalidGraphicsContext,
    cudaErrorNvlinkUncorrectable,
    cudaErrorJitCompilerNotFound,
    cudaErrorCooperativeLaunchTooLarge,
    cudaErrorStartupFailure,
    cudaErrorApiFailureBase,
    cudaChannelFormatKind,
    cudaChannelFormatKindSigned,
    cudaChannelFormatKindUnsigned,
    cudaChannelFormatKindFloat,
    cudaChannelFormatKindNone,
    cudaChannelFormatDesc,
    cudaArray_t,
    cudaArray_const_t,
    cudaMipmappedArray_t,
    cudaMipmappedArray_const_t,
    cudaMemoryType,
    cudaMemoryTypeHost,
    cudaMemoryTypeDevice,
    cudaMemcpyKind,
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDefault,
    cudaPitchedPtr,
    cudaExtent,
    cudaPos,
    cudaMemcpy3DParms,
    cudaMemcpy3DPeerParms,
    cudaGraphicsRegisterFlags,
    cudaGraphicsRegisterFlagsNone,
    cudaGraphicsRegisterFlagsReadOnly,
    cudaGraphicsRegisterFlagsWriteDiscard,
    cudaGraphicsRegisterFlagsSurfaceLoadStore,
    cudaGraphicsRegisterFlagsTextureGather,
    cudaGraphicsMapFlags,
    cudaGraphicsMapFlagsNone,
    cudaGraphicsMapFlagsReadOnly,
    cudaGraphicsMapFlagsWriteDiscard,
    cudaGraphicsCubeFace,
    cudaGraphicsCubeFacePositiveX,
    cudaGraphicsCubeFaceNegativeX,
    cudaGraphicsCubeFacePositiveY,
    cudaGraphicsCubeFaceNegativeY,
    cudaGraphicsCubeFacePositiveZ,
    cudaGraphicsCubeFaceNegativeZ,
    cudaResourceType,
    cudaResourceTypeArray,
    cudaResourceTypeMipmappedArray,
    cudaResourceTypeLinear,
    cudaResourceTypePitch2D,
    cudaResourceViewFormat,
    cudaResViewFormatNone,
    cudaResViewFormatUnsignedChar1,
    cudaResViewFormatUnsignedChar2,
    cudaResViewFormatUnsignedChar4,
    cudaResViewFormatSignedChar1,
    cudaResViewFormatSignedChar2,
    cudaResViewFormatSignedChar4,
    cudaResViewFormatUnsignedShort1,
    cudaResViewFormatUnsignedShort2,
    cudaResViewFormatUnsignedShort4,
    cudaResViewFormatSignedShort1,
    cudaResViewFormatSignedShort2,
    cudaResViewFormatSignedShort4,
    cudaResViewFormatUnsignedInt1,
    cudaResViewFormatUnsignedInt2,
    cudaResViewFormatUnsignedInt4,
    cudaResViewFormatSignedInt1,
    cudaResViewFormatSignedInt2,
    cudaResViewFormatSignedInt4,
    cudaResViewFormatHalf1,
    cudaResViewFormatHalf2,
    cudaResViewFormatHalf4,
    cudaResViewFormatFloat1,
    cudaResViewFormatFloat2,
    cudaResViewFormatFloat4,
    cudaResViewFormatUnsignedBlockCompressed1,
    cudaResViewFormatUnsignedBlockCompressed2,
    cudaResViewFormatUnsignedBlockCompressed3,
    cudaResViewFormatUnsignedBlockCompressed4,
    cudaResViewFormatSignedBlockCompressed4,
    cudaResViewFormatUnsignedBlockCompressed5,
    cudaResViewFormatSignedBlockCompressed5,
    cudaResViewFormatUnsignedBlockCompressed6H,
    cudaResViewFormatSignedBlockCompressed6H,
    cudaResViewFormatUnsignedBlockCompressed7,
    cudaResourceDesc_res_array,
    cudaResourceDesc_res_mipmap,
    cudaResourceDesc_res_linear,
    cudaResourceDesc_res_pitch2D,
    cudaResourceDesc_res,
    cudaResourceDesc,
    cudaResourceViewDesc,
    cudaPointerAttributes,
    cudaFuncAttributes,
    cudaFuncAttribute,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaFuncAttributeMax,
    cudaFuncCache,
    cudaFuncCachePreferNone,
    cudaFuncCachePreferShared,
    cudaFuncCachePreferL1,
    cudaFuncCachePreferEqual,
    cudaSharedMemConfig,
    cudaSharedMemBankSizeDefault,
    cudaSharedMemBankSizeFourByte,
    cudaSharedMemBankSizeEightByte,
    cudaSharedCarveout,
    cudaSharedmemCarveoutDefault,
    cudaSharedmemCarveoutMaxShared,
    cudaSharedmemCarveoutMaxL1,
    cudaComputeMode,
    cudaComputeModeDefault,
    cudaComputeModeExclusive,
    cudaComputeModeProhibited,
    cudaComputeModeExclusiveProcess,
    cudaLimit,
    cudaLimitStackSize,
    cudaLimitPrintfFifoSize,
    cudaLimitMallocHeapSize,
    cudaLimitDevRuntimeSyncDepth,
    cudaLimitDevRuntimePendingLaunchCount,
    cudaMemoryAdvise,
    cudaMemAdviseSetReadMostly,
    cudaMemAdviseUnsetReadMostly,
    cudaMemAdviseSetPreferredLocation,
    cudaMemAdviseUnsetPreferredLocation,
    cudaMemAdviseSetAccessedBy,
    cudaMemAdviseUnsetAccessedBy,
    cudaMemRangeAttribute,
    cudaMemRangeAttributeReadMostly,
    cudaMemRangeAttributePreferredLocation,
    cudaMemRangeAttributeAccessedBy,
    cudaMemRangeAttributeLastPrefetchLocation,
    cudaOutputMode_t,
    cudaKeyValuePair,
    cudaCSV,
    cudaDeviceAttr,
    cudaDevAttrMaxThreadsPerBlock,
    cudaDevAttrMaxBlockDimX,
    cudaDevAttrMaxBlockDimY,
    cudaDevAttrMaxBlockDimZ,
    cudaDevAttrMaxGridDimX,
    cudaDevAttrMaxGridDimY,
    cudaDevAttrMaxGridDimZ,
    cudaDevAttrMaxSharedMemoryPerBlock,
    cudaDevAttrTotalConstantMemory,
    cudaDevAttrWarpSize,
    cudaDevAttrMaxPitch,
    cudaDevAttrMaxRegistersPerBlock,
    cudaDevAttrClockRate,
    cudaDevAttrTextureAlignment,
    cudaDevAttrGpuOverlap,
    cudaDevAttrMultiProcessorCount,
    cudaDevAttrKernelExecTimeout,
    cudaDevAttrIntegrated,
    cudaDevAttrCanMapHostMemory,
    cudaDevAttrComputeMode,
    cudaDevAttrMaxTexture1DWidth,
    cudaDevAttrMaxTexture2DWidth,
    cudaDevAttrMaxTexture2DHeight,
    cudaDevAttrMaxTexture3DWidth,
    cudaDevAttrMaxTexture3DHeight,
    cudaDevAttrMaxTexture3DDepth,
    cudaDevAttrMaxTexture2DLayeredWidth,
    cudaDevAttrMaxTexture2DLayeredHeight,
    cudaDevAttrMaxTexture2DLayeredLayers,
    cudaDevAttrSurfaceAlignment,
    cudaDevAttrConcurrentKernels,
    cudaDevAttrEccEnabled,
    cudaDevAttrPciBusId,
    cudaDevAttrPciDeviceId,
    cudaDevAttrTccDriver,
    cudaDevAttrMemoryClockRate,
    cudaDevAttrGlobalMemoryBusWidth,
    cudaDevAttrL2CacheSize,
    cudaDevAttrMaxThreadsPerMultiProcessor,
    cudaDevAttrAsyncEngineCount,
    cudaDevAttrUnifiedAddressing,
    cudaDevAttrMaxTexture1DLayeredWidth,
    cudaDevAttrMaxTexture1DLayeredLayers,
    cudaDevAttrMaxTexture2DGatherWidth,
    cudaDevAttrMaxTexture2DGatherHeight,
    cudaDevAttrMaxTexture3DWidthAlt,
    cudaDevAttrMaxTexture3DHeightAlt,
    cudaDevAttrMaxTexture3DDepthAlt,
    cudaDevAttrPciDomainId,
    cudaDevAttrTexturePitchAlignment,
    cudaDevAttrMaxTextureCubemapWidth,
    cudaDevAttrMaxTextureCubemapLayeredWidth,
    cudaDevAttrMaxTextureCubemapLayeredLayers,
    cudaDevAttrMaxSurface1DWidth,
    cudaDevAttrMaxSurface2DWidth,
    cudaDevAttrMaxSurface2DHeight,
    cudaDevAttrMaxSurface3DWidth,
    cudaDevAttrMaxSurface3DHeight,
    cudaDevAttrMaxSurface3DDepth,
    cudaDevAttrMaxSurface1DLayeredWidth,
    cudaDevAttrMaxSurface1DLayeredLayers,
    cudaDevAttrMaxSurface2DLayeredWidth,
    cudaDevAttrMaxSurface2DLayeredHeight,
    cudaDevAttrMaxSurface2DLayeredLayers,
    cudaDevAttrMaxSurfaceCubemapWidth,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers,
    cudaDevAttrMaxTexture1DLinearWidth,
    cudaDevAttrMaxTexture2DLinearWidth,
    cudaDevAttrMaxTexture2DLinearHeight,
    cudaDevAttrMaxTexture2DLinearPitch,
    cudaDevAttrMaxTexture2DMipmappedWidth,
    cudaDevAttrMaxTexture2DMipmappedHeight,
    cudaDevAttrComputeCapabilityMajor,
    cudaDevAttrComputeCapabilityMinor,
    cudaDevAttrMaxTexture1DMipmappedWidth,
    cudaDevAttrStreamPrioritiesSupported,
    cudaDevAttrGlobalL1CacheSupported,
    cudaDevAttrLocalL1CacheSupported,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor,
    cudaDevAttrMaxRegistersPerMultiprocessor,
    cudaDevAttrManagedMemory,
    cudaDevAttrIsMultiGpuBoard,
    cudaDevAttrMultiGpuBoardGroupID,
    cudaDevAttrHostNativeAtomicSupported,
    cudaDevAttrSingleToDoublePrecisionPerfRatio,
    cudaDevAttrPageableMemoryAccess,
    cudaDevAttrConcurrentManagedAccess,
    cudaDevAttrComputePreemptionSupported,
    cudaDevAttrCanUseHostPointerForRegisteredMem,
    cudaDevAttrReserved92,
    cudaDevAttrReserved93,
    cudaDevAttrReserved94,
    cudaDevAttrCooperativeLaunch,
    cudaDevAttrCooperativeMultiDeviceLaunch,
    cudaDevAttrMaxSharedMemoryPerBlockOptin,
    cudaDeviceP2PAttr,
    cudaDevP2PAttrPerformanceRank,
    cudaDevP2PAttrAccessSupported,
    cudaDevP2PAttrNativeAtomicSupported,
    cudaDeviceProp,
    cudaDevicePropDontCare,
    CUDA_IPC_HANDLE_SIZE,
    cudaIpcEventHandle_t,
    cudaIpcMemHandle_t,
    cudaCGScope,
    cudaCGScopeInvalid,
    cudaCGScopeGrid,
    cudaCGScopeMultiGrid,
    cudaLaunchParams,
    cudaStreamCallback_t,
    cudaSurfaceType1D,
    cudaSurfaceType2D,
    cudaSurfaceType3D,
    cudaSurfaceTypeCubemap,
    cudaSurfaceType1DLayered,
    cudaSurfaceType2DLayered,
    cudaSurfaceTypeCubemapLayered,
    cudaSurfaceBoundaryMode,
    cudaBoundaryModeZero,
    cudaBoundaryModeClamp,
    cudaBoundaryModeTrap,
    cudaSurfaceFormatMode,
    cudaFormatModeForced,
    cudaFormatModeAuto,
    surfaceReference,
    cudaSurfaceObject_t,
    cudaTextureType1D,
    cudaTextureType2D,
    cudaTextureType3D,
    cudaTextureTypeCubemap,
    cudaTextureType1DLayered,
    cudaTextureType2DLayered,
    cudaTextureTypeCubemapLayered,
    cudaTextureAddressMode,
    cudaAddressModeWrap,
    cudaAddressModeClamp,
    cudaAddressModeMirror,
    cudaAddressModeBorder,
    cudaTextureFilterMode,
    cudaFilterModePoint,
    cudaFilterModeLinear,
    cudaTextureReadMode,
    cudaReadModeElementType,
    cudaReadModeNormalizedFloat,
    textureReference,
    cudaTextureDesc,
    cudaTextureObject_t,

    # functions
    cudaDeviceReset,
    cudaDeviceSynchronize,
    cudaDeviceSetLimit,
    cudaDeviceGetLimit,
    cudaDeviceGetCacheConfig,
    cudaDeviceGetStreamPriorityRange,
    cudaDeviceSetCacheConfig,
    cudaDeviceGetSharedMemConfig,
    cudaDeviceSetSharedMemConfig,
    cudaDeviceGetByPCIBusId,
    cudaDeviceGetPCIBusId,
    cudaIpcGetEventHandle,
    cudaIpcOpenEventHandle,
    cudaIpcGetMemHandle,
    cudaIpcOpenMemHandle,
    cudaIpcCloseMemHandle,
    cudaThreadExit,
    cudaThreadSynchronize,
    cudaThreadSetLimit,
    cudaThreadGetLimit,
    cudaThreadGetCacheConfig,
    cudaThreadSetCacheConfig,
    cudaGetLastError,
    cudaPeekAtLastError,
    cudaGetErrorName,
    cudaGetErrorString,
    cudaGetDeviceCount,
    cudaGetDeviceProperties,
    cudaDeviceGetAttribute,
    cudaDeviceGetP2PAttribute,
    cudaChooseDevice,
    cudaSetDevice,
    cudaGetDevice,
    cudaSetValidDevices,
    cudaSetDeviceFlags,
    cudaGetDeviceFlags,
    cudaStreamCreate,
    cudaStreamCreateWithFlags,
    cudaStreamCreateWithPriority,
    cudaStreamGetPriority,
    cudaStreamGetPriority_ptsz,
    cudaStreamGetFlags,
    cudaStreamGetFlags_ptsz,
    cudaStreamDestroy,
    cudaStreamWaitEvent,
    cudaStreamWaitEvent_ptsz,
    cudaStreamAddCallback,
    cudaStreamSynchronize,
    cudaStreamQuery,
    cudaStreamAttachMemAsync,
    cudaEventCreate,
    cudaEventCreateWithFlags,
    cudaEventRecord,
    cudaEventRecord_ptsz,
    cudaEventQuery,
    cudaEventSynchronize,
    cudaEventDestroy,
    cudaEventElapsedTime,
    cudaLaunchKernel,
    cudaLaunchKernel_ptsz,
    cudaLaunchCooperativeKernel,
    cudaLaunchCooperativeKernelMultiDevice,
    cudaFuncSetCacheConfig,
    cudaFuncSetSharedMemConfig,
    cudaFuncGetAttributes,
    cudaFuncSetAttribute,
    cudaSetDoubleForDevice,
    cudaSetDoubleForHost,
    cudaOccupancyMaxActiveBlocksPerMultiprocessor,
    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
    cudaConfigureCall,
    cudaSetupArgument,
    cudaLaunch,
    cudaLaunch_ptsz,
    cudaMallocManaged,
    cudaMalloc,
    cudaMallocHost,
    cudaMallocPitch,
    cudaMallocArray,
    cudaFree,
    cudaFreeHost,
    cudaFreeArray,
    cudaFreeMipmappedArray,
    cudaHostAlloc,
    cudaHostRegister,
    cudaHostUnregister,
    cudaHostGetDevicePointer,
    cudaHostGetFlags,
    cudaMalloc3D,
    cudaMalloc3DArray,
    cudaMallocMipmappedArray,
    cudaGetMipmappedArrayLevel,
    cudaMemcpy3D,
    cudaMemcpy3DPeer,
    cudaMemcpy3DAsync,
    cudaMemcpy3DAsync_ptsz,
    cudaMemcpy3DPeerAsync,
    cudaMemGetInfo,
    cudaArrayGetInfo,
    cudaMemcpy,
    cudaMemcpyPeer,
    cudaMemcpyToArray,
    cudaMemcpyFromArray,
    cudaMemcpyArrayToArray,
    cudaMemcpy2D,
    cudaMemcpy2DToArray,
    cudaMemcpy2DFromArray,
    cudaMemcpy2DArrayToArray,
    cudaMemcpyToSymbol,
    cudaMemcpyFromSymbol,
    cudaMemcpyAsync,
    cudaMemcpyAsync_ptsz,
    cudaMemcpyPeerAsync,
    cudaMemcpyToArrayAsync,
    cudaMemcpyFromArrayAsync,
    cudaMemcpy2DAsync,
    cudaMemcpy2DAsync_ptsz,
    cudaMemcpy2DToArrayAsync,
    cudaMemcpy2DFromArrayAsync,
    cudaMemcpyToSymbolAsync,
    cudaMemcpyFromSymbolAsync,
    cudaMemset,
    cudaMemset2D,
    cudaMemset3D,
    cudaMemsetAsync,
    cudaMemsetAsync_ptsz,
    cudaMemset2DAsync,
    cudaMemset2DAsync_ptsz,
    cudaMemset3DAsync,
    cudaMemset3DAsync_ptsz,
    cudaGetSymbolAddress,
    cudaGetSymbolSize,
    cudaMemPrefetchAsync,
    cudaMemAdvise,
    cudaMemRangeGetAttribute,
    cudaMemRangeGetAttributes,
    cudaPointerGetAttributes,
    cudaDeviceCanAccessPeer,
    cudaDeviceEnablePeerAccess,
    cudaDeviceDisablePeerAccess,
    cudaGraphicsUnregisterResource,
    cudaGraphicsResourceSetMapFlags,
    cudaGraphicsMapResources,
    cudaGraphicsUnmapResources,
    cudaGraphicsResourceGetMappedPointer,
    cudaGraphicsSubResourceGetMappedArray,
    cudaGraphicsResourceGetMappedMipmappedArray,
    cudaGetChannelDesc,
    cudaCreateChannelDesc,
    cudaBindTexture,
    cudaBindTexture2D,
    cudaBindTextureToArray,
    cudaBindTextureToMipmappedArray,
    cudaUnbindTexture,
    cudaGetTextureAlignmentOffset,
    cudaGetTextureReference,
    cudaBindSurfaceToArray,
    cudaGetSurfaceReference,
    cudaCreateTextureObject,
    cudaDestroyTextureObject,
    cudaGetTextureObjectResourceDesc,
    cudaGetTextureObjectTextureDesc,
    cudaGetTextureObjectResourceViewDesc,
    cudaCreateSurfaceObject,
    cudaDestroySurfaceObject,
    cudaGetSurfaceObjectResourceDesc,
    cudaDriverGetVersion,
    cudaRuntimeGetVersion,
    cudaGetExportTable
