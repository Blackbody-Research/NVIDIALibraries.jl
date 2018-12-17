#=*
* Load CUBLAS library
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

using Libdl
using Printf

if (Sys.iswindows())
	# find the most up to date version of CUDA Toolkit installed
	@assert (length(readdir("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\")) != 0)
	local latest::VersionNumber = reduce(max, map(VersionNumber, readdir("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\")))
	const libcublas = Libdl.find_library([@sprintf("cublas%i_%i%i", Sys.WORD_SIZE, latest.major, latest.minor)], [@sprintf("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v%i.%i\\bin", latest.major, latest.minor)])
end
