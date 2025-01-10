/**
 * @file spline2.inl
 * @author Jinyang Liu, Shixun Wu, Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-05-15
 *
 * (copyright to be updated)
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_KERNEL_SPLINE3_CUH
#define CUSZ_KERNEL_SPLINE3_CUH

#include <stdint.h>
#include <stdio.h>

#include <type_traits>

#include "cusz/type.h"
#include "utils/err.hh"
#include "utils/timer.hh"

#define SPLINE3_COMPR true
#define SPLINE3_DECOMPR false

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#define TIX threadIdx.x
#define TIY threadIdx.y
#define TIZ threadIdx.z
#define BIX blockIdx.x
#define BIY blockIdx.y
#define BIZ blockIdx.z
#define BDX blockDim.x
#define BDY blockDim.y
#define BDZ blockDim.z
#define GDX gridDim.x
#define GDY gridDim.y
#define GDZ gridDim.z

using DIM = u4;
using STRIDE = u4;
using DIM3 = dim3;
using STRIDE3 = dim3;

constexpr int BLOCK8 = 8;
constexpr int BLOCK32 = 32;
constexpr int DEFAULT_LINEAR_BLOCK_SIZE = 384;

#define SHM_ERROR s_ectrl

namespace cusz {

/********************************************************************************
 * host API
 ********************************************************************************/
template <typename TITER, int LINEAR_BLOCK_SIZE>
__global__ void c_spline2d_profiling_16x16x16data(
    TITER data, DIM3 data_size, STRIDE3 data_leap, TITER errors);

template <typename TITER, int LINEAR_BLOCK_SIZE>
__global__ void c_spline2d_profiling_data_2(
    TITER data, DIM3 data_size, STRIDE3 data_leap, TITER errors);

template <
    typename TITER, typename EITER, typename FP = float,
    int AnchorBlockSizeX = 8, int AnchorBlockSizeY = 8,
    int AnchorBlockSizeZ = 1,
    int numAnchorBlockX = 4,  // Number of Anchor blocks along X
    int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
    int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE,
    typename CompactVal = TITER, typename CompactIdx = uint32_t*,
    typename CompactNum = uint32_t*>
__global__ void c_spline2d_infprecis_data(
    TITER data, DIM3 data_size, STRIDE3 data_leap, EITER ectrl,
    DIM3 ectrl_size, STRIDE3 ectrl_leap, TITER anchor, STRIDE3 anchor_leap,
    CompactVal cval, CompactIdx cidx, CompactNum cn, FP eb_r, FP ebx2,
    int radius, INTERPOLATION_PARAMS intp_param, TITER errors);

template <
    typename EITER, typename TITER, typename FP = float,
    int AnchorBlockSizeX = 8, int AnchorBlockSizeY = 8,
    int AnchorBlockSizeZ = 1,
    int numAnchorBlockX = 4,  // Number of Anchor blocks along X
    int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
    int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__global__ void x_spline2d_infprecis_data(
    EITER ectrl,          // input 1
    DIM3 ectrl_size,      //
    STRIDE3 ectrl_leap,   //
    TITER anchor,         // input 2
    DIM3 anchor_size,     //
    STRIDE3 anchor_leap,  //
    TITER data,           // output
    DIM3 data_size,       //
    STRIDE3 data_leap,    //
    FP eb_r, FP ebx2, int radius, INTERPOLATION_PARAMS intp_param);

namespace device_api {
/********************************************************************************
 * device API
 ********************************************************************************/

template <typename T, int LINEAR_BLOCK_SIZE>
__device__ void auto_tuning_2d(
    volatile T s_data[9][9][33], volatile T local_errs[6], DIM3 data_size,
    volatile T* count);

template <typename T, int LINEAR_BLOCK_SIZE>
__device__ void auto_tuning_2_2d(
    volatile T s_data[9][9][33], volatile T local_errs[6], DIM3 data_size,
    volatile T* count);

template <
    typename T1, typename T2, typename FP, int AnchorBlockSizeX = 8,
    int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
    int numAnchorBlockX = 4,  // Number of Anchor blocks along X
    int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
    int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE, bool WORKFLOW = SPLINE3_COMPR,
    bool PROBE_PRED_ERROR = false>
__device__ void spline2d_layout2_interpolate(
    volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + 1]
                      [AnchorBlockSizeY * numAnchorBlockY + 1]
                      [AnchorBlockSizeX * numAnchorBlockX + 1],
    volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + 1]
                       [AnchorBlockSizeY * numAnchorBlockY + 1]
                       [AnchorBlockSizeX * numAnchorBlockX + 1],
    DIM3 data_size, FP eb_r, FP ebx2, int radius,
    INTERPOLATION_PARAMS intp_param);
}  // namespace device_api

}  // namespace cusz

/********************************************************************************
 * helper function
 ********************************************************************************/

namespace {

template <
    int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ,
    int numAnchorBlockX,  // Number of Anchor blocks along X
    int numAnchorBlockY,  // Number of Anchor blocks along Y
    int numAnchorBlockZ,  // Number of Anchor blocks along Z
    bool INCLUSIVE = true>
__forceinline__ __device__ bool xyz_predicate(
    unsigned int x, unsigned int y, unsigned int z, const DIM3& data_size)
{
  if CONSTEXPR (INCLUSIVE) {
    return (x <= (AnchorBlockSizeX * numAnchorBlockX) and
            y <= (AnchorBlockSizeY * numAnchorBlockY) and
            z <= (AnchorBlockSizeZ * numAnchorBlockZ)) and
           BIX * (AnchorBlockSizeX * numAnchorBlockX) + x < data_size.x and
           BIY * (AnchorBlockSizeY * numAnchorBlockY) + y < data_size.y and
           BIZ * (AnchorBlockSizeZ * numAnchorBlockZ) + z < data_size.z;
  }
  else {
    return x < (AnchorBlockSizeX * numAnchorBlockX) + (BIX == GDX - 1) and
           y < (AnchorBlockSizeY * numAnchorBlockY) + (BIY == GDY - 1) and
           z < (AnchorBlockSizeZ * numAnchorBlockZ) + (BIZ == GDZ - 1) and
           BIX * (AnchorBlockSizeX * numAnchorBlockX) + x < data_size.x and
           BIY * (AnchorBlockSizeY * numAnchorBlockY) + y < data_size.y and
           BIZ * (AnchorBlockSizeZ * numAnchorBlockZ) + z < data_size.z;
  }
}

// // control block_id3 in function call
// template <
//     typename T, bool PRINT_FP = true, int XEND = 33, int YEND = 9,
//     int ZEND = 9>
// __device__ void spline2d_print_block_from_GPU(
//     T volatile a[9][9][33], int radius = 512, bool compress = true,
//     bool print_ectrl = true)
// {
//   for (auto z = 0; z < ZEND; z++) {
//     printf("\nprint from GPU, z=%d\n", z);
//     printf("    ");
//     for (auto i = 0; i < 33; i++) printf("%3d", i);
//     printf("\n");

//     for (auto y = 0; y < YEND; y++) {
//       printf("y=%d ", y);
//       for (auto x = 0; x < XEND; x++) {  //
//         if CONSTEXPR (PRINT_FP) { printf("%.2e\t", (float)a[z][y][x]); }
//         else {
//           T c = print_ectrl ? a[z][y][x] - radius : a[z][y][x];
//           if (compress) {
//             if (c == 0) { printf("%3c", '.'); }
//             else {
//               if (abs(c) >= 10) { printf("%3c", '*'); }
//               else {
//                 if (print_ectrl) { printf("%3d", c); }
//                 else {
//                   printf("%4.2f", c);
//                 }
//               }
//             }
//           }
//           else {
//             if (print_ectrl) { printf("%3d", c); }
//             else {
//               printf("%4.2f", c);
//             }
//           }
//         }
//       }
//       printf("\n");
//     }
//   }
//   printf("\nGPU print end\n\n");
// }

template <
    typename T1, typename T2, int AnchorBlockSizeX, int AnchorBlockSizeY,
    int AnchorBlockSizeZ,
    int numAnchorBlockX,  // Number of Anchor blocks along X
    int numAnchorBlockY,  // Number of Anchor blocks along Y
    int numAnchorBlockZ,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void c_reset_scratch_data(
    volatile T1 s_data[1]
                      [AnchorBlockSizeY * numAnchorBlockY + 1]
                      [AnchorBlockSizeX * numAnchorBlockX + 1],
    volatile T2 s_ectrl[1]
                       [AnchorBlockSizeY * numAnchorBlockY + 1]
                       [AnchorBlockSizeX * numAnchorBlockX + 1],
    int radius)
{
  // alternatively, reinterprete cast volatile T?[][][] to 1D
  for (auto _tix = TIX; _tix < (AnchorBlockSizeX * numAnchorBlockX + 1) *
                                   (AnchorBlockSizeY * numAnchorBlockY + 1);
       _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + 1));
    auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + 1)) %
             (AnchorBlockSizeY * numAnchorBlockY + 1);
    auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + 1)) /
             (AnchorBlockSizeY * numAnchorBlockY + 1);

    s_data[z][y][x] = 0;
    /*****************************************************************************
     okay to use
     ******************************************************************************/
    if (x % AnchorBlockSizeX == 0 and y % AnchorBlockSizeY == 0 and
        z % AnchorBlockSizeZ == 0)
      s_ectrl[z][y][x] = radius;
    /*****************************************************************************
     alternatively
     ******************************************************************************/
    // s_ectrl[z][y][x] = radius;
  }
  __syncthreads();
}

template <typename T, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void c_reset_scratch_profiling_16x16x16data(
    volatile T s_data[16][16][16], T default_value)
{
  for (auto _tix = TIX; _tix < 16 * 16 * 16; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % 16);
    auto y = (_tix / 16) % 16;
    auto z = (_tix / 16) / 16;

    s_data[z][y][x] = default_value;
  }
}

template <typename T, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void c_reset_scratch_profiling_data_2(
    volatile T s_data[64], T nx[64][4], T ny[64][4], T nz[64][4],
    T default_value)
{
  for (auto _tix = TIX; _tix < 64 * 4; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % 4);
    auto yz = _tix / 4;

    nx[yz][x] = ny[yz][x] = nz[yz][x] = default_value;
    s_data[TIX] = default_value;
  }
}

template <
    typename T1, int AnchorBlockSizeX, int AnchorBlockSizeY,
    int AnchorBlockSizeZ,
    int numAnchorBlockX,  // Number of Anchor blocks along X
    int numAnchorBlockY,  // Number of Anchor blocks along Y
    int numAnchorBlockZ,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void c_gather_anchor(
    T1* data, DIM3 data_size, STRIDE3 data_leap, T1* anchor,
    STRIDE3 anchor_leap)
{
  auto x = (TIX % (AnchorBlockSizeX * numAnchorBlockX)) +
           BIX * (AnchorBlockSizeX * numAnchorBlockX);
  auto y = (TIX / (AnchorBlockSizeX * numAnchorBlockX)) %
               (AnchorBlockSizeY * numAnchorBlockY) +
           BIY * (AnchorBlockSizeY * numAnchorBlockY);
  auto z = (TIX / (AnchorBlockSizeX * numAnchorBlockX)) /
               (AnchorBlockSizeY * numAnchorBlockY) +
           BIZ * (AnchorBlockSizeZ * numAnchorBlockZ);

  bool pred1 = x % AnchorBlockSizeX == 0 and y % AnchorBlockSizeY == 0 and
               z % AnchorBlockSizeZ == 0;
  bool pred2 = x < data_size.x and y < data_size.y and z < data_size.z;

  if (pred1 and pred2) {
    auto data_id = x + y * data_leap.y + z * data_leap.z;
    auto anchor_id = (x / AnchorBlockSizeX) +
                     (y / AnchorBlockSizeY) * anchor_leap.y +
                     (z / AnchorBlockSizeZ) * anchor_leap.z;
    anchor[anchor_id] = data[data_id];
  }
  __syncthreads();
}

template <
    typename T1, typename T2 = T1, int AnchorBlockSizeX = 8,
    int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
    int numAnchorBlockX = 4,  // Number of Anchor blocks along X
    int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
    int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void x_reset_scratch_data(
    volatile T1 s_xdata[1]
                       [AnchorBlockSizeY * numAnchorBlockY + 1]
                       [AnchorBlockSizeX * numAnchorBlockX + 1],
    volatile T2 s_ectrl[1]
                       [AnchorBlockSizeY * numAnchorBlockY + 1]
                       [AnchorBlockSizeX * numAnchorBlockX + 1],
    T1* anchor, DIM3 anchor_size, STRIDE3 anchor_leap)
{
  for (auto _tix = TIX; _tix < (AnchorBlockSizeX * numAnchorBlockX + 1) *
                                   (AnchorBlockSizeY * numAnchorBlockY + 1);
       _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + 1));
    auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + 1)) %
             (AnchorBlockSizeY * numAnchorBlockY + 1);
    auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + 1)) /
             (AnchorBlockSizeY * numAnchorBlockY + 1);

    s_ectrl[z][y][x] = 0;  // TODO explicitly handle zero-padding
    /*****************************************************************************
     okay to use
     ******************************************************************************/
    // Todo 2d
    // Here 8 is the interpolation block size, not the entire compression
    // manipulated by a threadblock, need to dinstiguish with CompressionBlock.
    if (x % AnchorBlockSizeX == 0 and y % AnchorBlockSizeY == 0 and
        z % AnchorBlockSizeZ == 0) {
      s_xdata[z][y][x] = 0;

      auto ax = ((x / AnchorBlockSizeX) + BIX * numAnchorBlockX);
      auto ay = ((y / AnchorBlockSizeY) + BIY * numAnchorBlockY);
      auto az = ((z / AnchorBlockSizeZ) + BIZ * numAnchorBlockZ);

      if (ax < anchor_size.x and ay < anchor_size.y and az < anchor_size.z)
        s_xdata[z][y][x] =
            anchor[ax + ay * anchor_leap.y + az * anchor_leap.z];
    }
    /*****************************************************************************
     alternatively
     ******************************************************************************/
  }

  __syncthreads();
}

template <
    typename T1, typename T2, int AnchorBlockSizeX = 8,
    int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
    int numAnchorBlockX = 4,  // Number of Anchor blocks along X
    int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
    int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_data(
    T1* data, DIM3 data_size, STRIDE3 data_leap,
    volatile T2 s_data[1]
                      [AnchorBlockSizeY * numAnchorBlockY + 1]
                      [AnchorBlockSizeX * numAnchorBlockX + 1])
{
  constexpr auto TOTAL = (AnchorBlockSizeX * numAnchorBlockX + 1) *
                         (AnchorBlockSizeY * numAnchorBlockY + 1);
  // if(TIX + TIY + TIZ == 0 && BIX + BIY + BIZ == 0) printf(" data_leap=%d %d %d\n",  data_leap.x,  data_leap.y,  data_leap.z);
  for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + 1));
    auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + 1)) %
             (AnchorBlockSizeY * numAnchorBlockY + 1);
    auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + 1)) /
             (AnchorBlockSizeY * numAnchorBlockY + 1);
    auto gx = (x + BIX * (AnchorBlockSizeX * numAnchorBlockX));
    auto gy = (y + BIY * (AnchorBlockSizeY * numAnchorBlockY));
    auto gz = (z + BIZ * (AnchorBlockSizeZ * numAnchorBlockZ));
    auto gid = gx + gy * data_leap.y + gz * data_leap.z;
   
    if (gx < data_size.x and gy < data_size.y and gz < data_size.z)
      s_data[z][y][x] = data[gid];
    // if(BIX + BIY + BIZ == 0)
    // printf(" block %d %d %d, thread %d %d %d gid=%d -->sx, sy, sz = %d, %d, %d: data=%f\n",
    // BIX, BIY, BIZ, TIX, TIY, TIZ, gid, x, y, z, data[gid]);

  }
  __syncthreads();
}

template <
    typename T1, typename T2,
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_profiling_16x16x16data(
    T1* data, DIM3 data_size, STRIDE3 data_leap,
    volatile T2 s_data[16][16][16])
{
  constexpr auto TOTAL = 16 * 16 * 16;

  for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % 16);
    auto y = (_tix / 16) % 16;
    auto z = (_tix / 16) / 16;
    auto gx_1 = x / 4;
    auto gx_2 = x % 4;
    auto gy_1 = y / 4;
    auto gy_2 = y % 4;
    auto gz_1 = z / 4;
    auto gz_2 = z % 4;
    auto gx = (data_size.x / 4) * gx_1 + gx_2;
    auto gy = (data_size.y / 4) * gy_1 + gy_2;
    auto gz = (data_size.z / 4) * gz_1 + gz_2;

    auto gid = gx + gy * data_leap.y + gz * data_leap.z;

    if (gx < data_size.x and gy < data_size.y and gz < data_size.z)
      s_data[z][y][x] = data[gid];
  }
  __syncthreads();
}

template <
    typename T1, typename T2,
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_profiling_data_2(
    T1* data, DIM3 data_size, STRIDE3 data_leap, volatile T2 s_data[64],
    volatile T2 s_nx[64][4], volatile T2 s_ny[64][4], volatile T2 s_nz[64][4])
{
  constexpr auto TOTAL = 64 * 4;
  int factors[4] = {-3, -1, 1, 3};
  for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
    auto offset = (_tix % 4);
    auto idx = _tix / 4;
    auto x = idx % 4;
    auto y = (idx / 4) % 4;
    auto z = (idx / 4) / 4;
    auto gx = (data_size.x / 4) * x + data_size.x / 8;
    auto gy = (data_size.y / 4) * y + data_size.y / 8;
    auto gz = (data_size.z / 4) * z + data_size.z / 8;

    auto gid = gx + gy * data_leap.y + gz * data_leap.z;

    if (gx >= 3 and gy >= 3 and gz >= 3 and gx + 3 < data_size.x and
        gy + 3 < data_size.y and gz + 3 < data_size.z) {
      s_data[idx] = data[gid];

      auto factor = factors[offset];
      s_nx[idx][offset] = data[gid + factor];
      s_ny[idx][offset] = data[gid + factor * data_leap.y];
      s_nz[idx][offset] = data[gid + factor * data_leap.z];
    }
  }
  __syncthreads();
}

template <
    typename T = float, typename E = u4, int AnchorBlockSizeX = 8,
    int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
    int numAnchorBlockX = 4,  // Number of Anchor blocks along X
    int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
    int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_fuse(
    E* ectrl, dim3 ectrl_size, dim3 ectrl_leap, T* scattered_outlier,
    volatile T s_ectrl[1]
                      [AnchorBlockSizeY * numAnchorBlockY + 1]
                      [AnchorBlockSizeX * numAnchorBlockX + 1])
{
  constexpr auto TOTAL = (AnchorBlockSizeX * numAnchorBlockX + 1) *
                         (AnchorBlockSizeY * numAnchorBlockY + 1);

  for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + 1));
    auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + 1)) %
             (AnchorBlockSizeY * numAnchorBlockY + 1);
    auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + 1)) /
             (AnchorBlockSizeY * numAnchorBlockY + 1);
    auto gx = (x + BIX * (AnchorBlockSizeX * numAnchorBlockX));
    auto gy = (y + BIY * (AnchorBlockSizeY * numAnchorBlockY));
    auto gz = (z + BIZ * (AnchorBlockSizeZ * numAnchorBlockZ));
    // if(TIX + TIY + TIZ == 0 && BIX + BIY + BIZ == 0) printf(" ectrl_leap=%d %d %d\n",  ectrl_leap.x,  ectrl_leap.y,  ectrl_leap.z);
    auto gid = gx + gy * ectrl_leap.y + gz * ectrl_leap.z;

    if (gx < ectrl_size.x and gy < ectrl_size.y and gz < ectrl_size.z)
      s_ectrl[z][y][x] = static_cast<T>(ectrl[gid]) + scattered_outlier[gid];
  }
  __syncthreads();
}

// dram_outlier should be the same in type with shared memory buf
template <
    typename T1, typename T2, int AnchorBlockSizeX = 8,
    int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
    int numAnchorBlockX = 4,  // Number of Anchor blocks along X
    int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
    int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void shmem2global_data(
    volatile T1 s_buf[1]
                     [AnchorBlockSizeY * numAnchorBlockY + 1]
                     [AnchorBlockSizeX * numAnchorBlockX + 1],
    T2* dram_buf, DIM3 buf_size, STRIDE3 buf_leap)
{
  auto x_size = AnchorBlockSizeX * numAnchorBlockX + (BIX == GDX - 1);
  auto y_size = AnchorBlockSizeY * numAnchorBlockY + (BIY == GDY - 1);
  auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (BIZ == GDZ - 1);
  // constexpr auto TOTAL = 32 * 8 * 8;
  auto TOTAL = x_size * y_size;

  for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % x_size);
    auto y = (_tix / x_size) % y_size;
    auto z = (_tix / x_size) / y_size;
    auto gx = (x + BIX * AnchorBlockSizeX * numAnchorBlockX);
    auto gy = (y + BIY * AnchorBlockSizeY * numAnchorBlockY);
    auto gz = (z + BIZ * AnchorBlockSizeZ * numAnchorBlockZ);
    auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;

    if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z)
      dram_buf[gid] = s_buf[z][y][x];
    // if(BIX + BIY + BIZ == 0) printf("gid=%d data=%f\n", gid, dram_buf[gid]);
  }
  __syncthreads();
}

// dram_outlier should be the same in type with shared memory buf
template <
    typename T1, typename T2, int AnchorBlockSizeX = 8,
    int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
    int numAnchorBlockX = 4,  // Number of Anchor blocks along X
    int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
    int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void shmem2global_data_with_compaction(
    volatile T1 s_buf[1]
                     [AnchorBlockSizeY * numAnchorBlockY + 1]
                     [AnchorBlockSizeX * numAnchorBlockX + 1],
    T2* dram_buf, DIM3 buf_size, STRIDE3 buf_leap, int radius,
    T1* dram_compactval = nullptr, uint32_t* dram_compactidx = nullptr,
    uint32_t* dram_compactnum = nullptr)
{
  auto x_size = AnchorBlockSizeX * numAnchorBlockX + (BIX == GDX - 1);
  auto y_size = AnchorBlockSizeY * numAnchorBlockY + (BIY == GDY - 1);
  auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (BIZ == GDZ - 1);
  auto TOTAL = x_size * y_size;

  for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % x_size);
    auto y = (_tix / x_size) % y_size;
    auto z = (_tix / x_size) / y_size;
    auto gx = (x + BIX * AnchorBlockSizeX * numAnchorBlockX);
    auto gy = (y + BIY * AnchorBlockSizeY * numAnchorBlockY);
    auto gz = (z + BIZ * AnchorBlockSizeZ * numAnchorBlockZ);
    auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;

    auto candidate = s_buf[z][y][x];
    bool quantizable = (candidate >= 0) and (candidate < 2 * radius);

    if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) {
      // TODO this is for algorithmic demo by reading from shmem
      // For performance purpose, it can be inlined in quantization
      dram_buf[gid] = quantizable * static_cast<T2>(candidate);

      if (not quantizable) {
        auto cur_idx = atomicAdd(dram_compactnum, 1);
        dram_compactidx[cur_idx] = gid;
        dram_compactval[cur_idx] = candidate;
      }
    }
  }
  __syncthreads();
}

template <
    typename T1, typename T2, typename FP, int AnchorBlockSizeX,
    int AnchorBlockSizeY, int AnchorBlockSizeZ,
    int numAnchorBlockX,  // Number of Anchor blocks along X
    int numAnchorBlockY,  // Number of Anchor blocks along Y
    int numAnchorBlockZ,  // Number of Anchor blocks along Z
    typename LAMBDAX, typename LAMBDAY, typename LAMBDAZ, bool BLUE,
    bool YELLOW, bool HOLLOW, int LINEAR_BLOCK_SIZE, int BLOCK_DIMX,
    int BLOCK_DIMY, bool COARSEN, int BLOCK_DIMZ, bool BORDER_INCLUSIVE,
    bool WORKFLOW>
__forceinline__ __device__ void interpolate_stage(
    volatile T1 s_data[1]
                      [AnchorBlockSizeY * numAnchorBlockY + 1]
                      [AnchorBlockSizeX * numAnchorBlockX + 1],
    volatile T2 s_ectrl[1]
                       [AnchorBlockSizeY * numAnchorBlockY + 1]
                       [AnchorBlockSizeX * numAnchorBlockX + 1],
    DIM3 data_size, LAMBDAX xmap, LAMBDAY ymap, LAMBDAZ zmap, int unit,
    FP eb_r, FP ebx2, int radius, bool interpolator)
{
  // static_assert(
  //     BLOCK_DIMX * BLOCK_DIMY * (COARSEN ? 1 : BLOCK_DIMZ) <= 384,
  //     "block oversized");
  static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
  static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
  static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
  static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");

  auto run = [&](auto x, auto y, auto z) {
    if (xyz_predicate<
            AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ,
            BORDER_INCLUSIVE>(x, y, z, data_size)) {
      T1 pred = 0;
      auto global_x = BIX * (AnchorBlockSizeX * numAnchorBlockX) + x;
      auto global_y = BIY * (AnchorBlockSizeY * numAnchorBlockY) + y;
      auto global_z = BIZ * (AnchorBlockSizeZ * numAnchorBlockZ) + z;
      if (interpolator == 0) {
        if CONSTEXPR (BLUE) {  //
          if (BIZ != GDZ - 1) {
            if (z >= 3 * unit and
                z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ)
              pred =
                  (-s_data[z - 3 * unit][y][x] + 9 * s_data[z - unit][y][x] +
                   9 * s_data[z + unit][y][x] - s_data[z + 3 * unit][y][x]) /
                  16;
            else if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ)
              pred = (3 * s_data[z - unit][y][x] + 6 * s_data[z + unit][y][x] -
                      s_data[z + 3 * unit][y][x]) /
                     8;
            else if (z >= 3 * unit)
              pred =
                  (-s_data[z - 3 * unit][y][x] + 6 * s_data[z - unit][y][x] +
                   3 * s_data[z + unit][y][x]) /
                  8;

            else
              pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
          }
          else {
            if (z >= 3 * unit) {
              if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ and
                  global_z + 3 * unit < data_size.z)
                pred =
                    (-s_data[z - 3 * unit][y][x] + 9 * s_data[z - unit][y][x] +
                     9 * s_data[z + unit][y][x] - s_data[z + 3 * unit][y][x]) /
                    16;
              else if (global_z + unit < data_size.z)
                pred =
                    (-s_data[z - 3 * unit][y][x] + 6 * s_data[z - unit][y][x] +
                     3 * s_data[z + unit][y][x]) /
                    8;
              else
                pred = s_data[z - unit][y][x];
            }
            else {
              if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ and
                  global_z + 3 * unit < data_size.z)
                pred =
                    (3 * s_data[z - unit][y][x] + 6 * s_data[z + unit][y][x] -
                     s_data[z + 3 * unit][y][x]) /
                    8;
              else if (global_z + unit < data_size.z)
                pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
              else
                pred = s_data[z - unit][y][x];
            }
          }
        }
        if CONSTEXPR (YELLOW) {  //
          if (BIY != GDY - 1) {
            if (y >= 3 * unit and
                y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY)
              pred =
                  (-s_data[z][y - 3 * unit][x] + 9 * s_data[z][y - unit][x] +
                   9 * s_data[z][y + unit][x] - s_data[z][y + 3 * unit][x]) /
                  16;
            else if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY)
              pred = (3 * s_data[z][y - unit][x] + 6 * s_data[z][y + unit][x] -
                      s_data[z][y + 3 * unit][x]) /
                     8;
            else if (y >= 3 * unit)
              pred =
                  (-s_data[z][y - 3 * unit][x] + 6 * s_data[z][y - unit][x] +
                   3 * s_data[z][y + unit][x]) /
                  8;
            else
              pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
          }
          else {
            if (y >= 3 * unit) {
              if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY and
                  global_y + 3 * unit < data_size.y)
                pred =
                    (-s_data[z][y - 3 * unit][x] + 9 * s_data[z][y - unit][x] +
                     9 * s_data[z][y + unit][x] - s_data[z][y + 3 * unit][x]) /
                    16;
              else if (global_y + unit < data_size.y)
                pred =
                    (-s_data[z][y - 3 * unit][x] + 6 * s_data[z][y - unit][x] +
                     3 * s_data[z][y + unit][x]) /
                    8;
              else
                pred = s_data[z][y - unit][x];
            }
            else {
              if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY and
                  global_y + 3 * unit < data_size.y)
                pred =
                    (3 * s_data[z][y - unit][x] + 6 * s_data[z][y + unit][x] -
                     s_data[z][y + 3 * unit][x]) /
                    8;
              else if (global_y + unit < data_size.y)
                pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
              else
                pred = s_data[z][y - unit][x];
            }
          }
        }

        if CONSTEXPR (HOLLOW) {  //
          if (BIX != GDX - 1) {
            if (x >= 3 * unit and
                x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX)
              pred =
                  (-s_data[z][y][x - 3 * unit] + 9 * s_data[z][y][x - unit] +
                   9 * s_data[z][y][x + unit] - s_data[z][y][x + 3 * unit]) /
                  16;
            else if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX)
              pred = (3 * s_data[z][y][x - unit] + 6 * s_data[z][y][x + unit] -
                      s_data[z][y][x + 3 * unit]) /
                     8;
            else if (x >= 3 * unit)
              pred =
                  (-s_data[z][y][x - 3 * unit] + 6 * s_data[z][y][x - unit] +
                   3 * s_data[z][y][x + unit]) /
                  8;
            else
              pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
          }
          else {
            if (x >= 3 * unit) {
              if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX and
                  global_x + 3 * unit < data_size.x)
                pred =
                    (-s_data[z][y][x - 3 * unit] + 9 * s_data[z][y][x - unit] +
                     9 * s_data[z][y][x + unit] - s_data[z][y][x + 3 * unit]) /
                    16;
              else if (global_x + unit < data_size.x)
                pred =
                    (-s_data[z][y][x - 3 * unit] + 6 * s_data[z][y][x - unit] +
                     3 * s_data[z][y][x + unit]) /
                    8;
              else
                pred = s_data[z][y][x - unit];
            }
            else {
              if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX and
                  global_x + 3 * unit < data_size.x)
                pred =
                    (3 * s_data[z][y][x - unit] + 6 * s_data[z][y][x + unit] -
                     s_data[z][y][x + 3 * unit]) /
                    8;
              else if (global_x + unit < data_size.x)
                pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
              else
                pred = s_data[z][y][x - unit];
            }
          }
        }
      }
      else {
        if CONSTEXPR (BLUE) {  //

          if (BIZ != GDZ - 1) {
            if (z >= 3 * unit and
                z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ)
              pred =
                  (-3 * s_data[z - 3 * unit][y][x] +
                   23 * s_data[z - unit][y][x] + 23 * s_data[z + unit][y][x] -
                   3 * s_data[z + 3 * unit][y][x]) /
                  40;
            else if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ)
              pred = (3 * s_data[z - unit][y][x] + 6 * s_data[z + unit][y][x] -
                      s_data[z + 3 * unit][y][x]) /
                     8;
            else if (z >= 3 * unit)
              pred =
                  (-s_data[z - 3 * unit][y][x] + 6 * s_data[z - unit][y][x] +
                   3 * s_data[z + unit][y][x]) /
                  8;

            else
              pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
          }
          else {
            if (z >= 3 * unit) {
              if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ and
                  global_z + 3 * unit < data_size.z)
                pred = (-3 * s_data[z - 3 * unit][y][x] +
                        23 * s_data[z - unit][y][x] +
                        23 * s_data[z + unit][y][x] -
                        3 * s_data[z + 3 * unit][y][x]) /
                       40;
              else if (global_z + unit < data_size.z)
                pred =
                    (-s_data[z - 3 * unit][y][x] + 6 * s_data[z - unit][y][x] +
                     3 * s_data[z + unit][y][x]) /
                    8;
              else
                pred = s_data[z - unit][y][x];
            }
            else {
              if (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ and
                  global_z + 3 * unit < data_size.z)
                pred =
                    (3 * s_data[z - unit][y][x] + 6 * s_data[z + unit][y][x] -
                     s_data[z + 3 * unit][y][x]) /
                    8;
              else if (global_z + unit < data_size.z)
                pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
              else
                pred = s_data[z - unit][y][x];
            }
          }
        }
        if CONSTEXPR (YELLOW) {  //
          if (BIY != GDY - 1) {
            if (y >= 3 * unit and
                y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY)
              pred =
                  (-3 * s_data[z][y - 3 * unit][x] +
                   23 * s_data[z][y - unit][x] + 23 * s_data[z][y + unit][x] -
                   3 * s_data[z][y + 3 * unit][x]) /
                  40;
            else if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY)
              pred = (3 * s_data[z][y - unit][x] + 6 * s_data[z][y + unit][x] -
                      s_data[z][y + 3 * unit][x]) /
                     8;
            else if (y >= 3 * unit)
              pred =
                  (-s_data[z][y - 3 * unit][x] + 6 * s_data[z][y - unit][x] +
                   3 * s_data[z][y + unit][x]) /
                  8;
            else
              pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
          }
          else {
            if (y >= 3 * unit) {
              if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY and
                  global_y + 3 * unit < data_size.y)
                pred = (-3 * s_data[z][y - 3 * unit][x] +
                        23 * s_data[z][y - unit][x] +
                        23 * s_data[z][y + unit][x] -
                        3 * s_data[z][y + 3 * unit][x]) /
                       40;
              else if (global_y + unit < data_size.y)
                pred =
                    (-s_data[z][y - 3 * unit][x] + 6 * s_data[z][y - unit][x] +
                     3 * s_data[z][y + unit][x]) /
                    8;
              else
                pred = s_data[z][y - unit][x];
            }
            else {
              if (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY and
                  global_y + 3 * unit < data_size.y)
                pred =
                    (3 * s_data[z][y - unit][x] + 6 * s_data[z][y + unit][x] -
                     s_data[z][y + 3 * unit][x]) /
                    8;
              else if (global_y + unit < data_size.y)
                pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
              else
                pred = s_data[z][y - unit][x];
            }
          }
        }

        if CONSTEXPR (HOLLOW) {  //
          if (BIX != GDX - 1) {
            if (x >= 3 * unit and
                x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX)
              pred =
                  (-3 * s_data[z][y][x - 3 * unit] +
                   23 * s_data[z][y][x - unit] + 23 * s_data[z][y][x + unit] -
                   3 * s_data[z][y][x + 3 * unit]) /
                  40;
            else if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX)
              pred = (3 * s_data[z][y][x - unit] + 6 * s_data[z][y][x + unit] -
                      s_data[z][y][x + 3 * unit]) /
                     8;
            else if (x >= 3 * unit)
              pred =
                  (-s_data[z][y][x - 3 * unit] + 6 * s_data[z][y][x - unit] +
                   3 * s_data[z][y][x + unit]) /
                  8;
            else
              pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
          }
          else {
            if (x >= 3 * unit) {
              if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX and
                  global_x + 3 * unit < data_size.x)
                pred = (-3 * s_data[z][y][x - 3 * unit] +
                        23 * s_data[z][y][x - unit] +
                        23 * s_data[z][y][x + unit] -
                        3 * s_data[z][y][x + 3 * unit]) /
                       40;
              else if (global_x + unit < data_size.x)
                pred =
                    (-s_data[z][y][x - 3 * unit] + 6 * s_data[z][y][x - unit] +
                     3 * s_data[z][y][x + unit]) /
                    8;
              else
                pred = s_data[z][y][x - unit];
            }
            else {
              if (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX and
                  global_x + 3 * unit < data_size.x)
                pred =
                    (3 * s_data[z][y][x - unit] + 6 * s_data[z][y][x + unit] -
                     s_data[z][y][x + 3 * unit]) /
                    8;
              else if (global_x + unit < data_size.x)
                pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
              else
                pred = s_data[z][y][x - unit];
            }
          }
        }
      }
      if CONSTEXPR (WORKFLOW == SPLINE3_COMPR) {
        auto err = s_data[z][y][x] - pred;
        decltype(err) code;
        // TODO unsafe, did not deal with the out-of-cap case
        {
          code = fabs(err) * eb_r + 1;
          code = err < 0 ? -code : code;
          code = int(code / 2) + radius;
        }
        // if()
        s_ectrl[z][y][x] = code;  // TODO double check if unsigned type works
        // if(BIX + BIY + BIZ == 0)
        // printf("TIX %d xyz %d %d %d, org=%f -->  pred=%f --> lossy=%f, code=%d\n", TIX, x, y, z, s_data[z][y][x],  pred, pred + (code - radius) * ebx2, code);
        s_data[z][y][x] = pred + (code - radius) * ebx2;
      }
      else {  // TODO == DECOMPRESSS and static_assert
        auto code = s_ectrl[z][y][x];
        // if(BIX + BIY + BIZ == 0)
        // printf("xyz %d %d %d, org=%f -->  pred=%f --> lossy=%f, code=%d\n",x, y, z, s_data[z][y][x],  pred, pred + (code - radius) * ebx2, code);
        s_data[z][y][x] = pred + (code - radius) * ebx2;
        
      }
    }
  };
  // if CONSTEXPR (COARSEN) {
    constexpr auto TOTAL = BLOCK_DIMX * BLOCK_DIMY;
    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
      auto itix = (_tix % BLOCK_DIMX);
      auto itiy = (_tix / BLOCK_DIMX) % BLOCK_DIMY;
      auto itiz = (_tix / BLOCK_DIMX) / BLOCK_DIMY;
      auto x = xmap(itix, unit);
      auto y = ymap(itiy, unit);
      auto z = zmap(itiz, unit);
      run(x, y, z);
    }
  // }
  // else {
  //   auto itix = (TIX % BLOCK_DIMX);
  //   auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
  //   auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
  //   auto x = xmap(itix, unit);
  //   auto y = ymap(itiy, unit);
  //   auto z = zmap(itiz, unit);
  //   run(x, y, z);
  // }
  __syncthreads();
}

}  // namespace
template <typename T, int LINEAR_BLOCK_SIZE>
__device__ void cusz::device_api::auto_tuning_2d(
    volatile T s_data[16][16][16], volatile T local_errs[2], DIM3 data_size,
    T* errs)
{
  if (TIX < 2) local_errs[TIX] = 0;
  __syncthreads();

  auto local_idx = TIX % 2;
  auto temp = TIX / 2;

  auto block_idx_x = temp % 4;
  auto block_idx_y = (temp / 4) % 4;
  auto block_idx_z = ((temp / 4) / 4) % 4;
  auto dir = ((temp / 4) / 4) / 4;
  bool predicate = dir < 2;
  if (predicate) {
    auto x = 4 * block_idx_x + 1 + local_idx;
    // auto x =16;
    auto y = 4 * block_idx_y + 1 + local_idx;
    auto z = 4 * block_idx_z + 1 + local_idx;

    T pred = 0;

    // auto unit = 1;
    switch (dir) {
      case 0: pred = (s_data[z - 1][y][x] + s_data[z + 1][y][x]) / 2; break;

      case 1: pred = (s_data[z][y][x - 1] + s_data[z][y][x + 1]) / 2; break;

      default: break;
    }

    T abs_error = fabs(pred - s_data[z][y][x]);
    atomicAdd(const_cast<T*>(local_errs) + dir, abs_error);
  }
  __syncthreads();
  if (TIX < 2) errs[TIX] = local_errs[TIX];
  __syncthreads();
}

template <typename T, int LINEAR_BLOCK_SIZE>
__device__ void cusz::device_api::auto_tuning_2_2d(
    volatile T s_data[64], volatile T s_nx[64][4], volatile T s_ny[64][4],
    volatile T s_nz[64][4], volatile T local_errs[6], DIM3 data_size, T* errs)
{
  if (TIX < 6) local_errs[TIX] = 0;
  __syncthreads();

  auto point_idx = TIX % 64;
  auto c = TIX / 64;

  bool predicate = c < 6;
  if (predicate) {
    T pred = 0;
    switch (c) {
      case 0:
        pred = (-s_nz[point_idx][0] + 9 * s_nz[point_idx][1] +
                9 * s_nz[point_idx][2] - s_nz[point_idx][3]) /
               16;
        break;

      case 1:
        pred = (-3 * s_nz[point_idx][0] + 23 * s_nz[point_idx][1] +
                23 * s_nz[point_idx][2] - 3 * s_nz[point_idx][3]) /
               40;
        break;
      case 2:
        pred = (-s_ny[point_idx][0] + 9 * s_ny[point_idx][1] +
                9 * s_ny[point_idx][2] - s_ny[point_idx][3]) /
               16;
        break;
      case 3:
        pred = (-3 * s_ny[point_idx][0] + 23 * s_ny[point_idx][1] +
                23 * s_ny[point_idx][2] - 3 * s_ny[point_idx][3]) /
               40;
        break;

      case 4:
        pred = (-s_nx[point_idx][0] + 9 * s_nx[point_idx][1] +
                9 * s_nx[point_idx][2] - s_nx[point_idx][3]) /
               16;
        break;
      case 5:
        pred = (-3 * s_nx[point_idx][0] + 23 * s_nx[point_idx][1] +
                23 * s_nx[point_idx][2] - 3 * s_nx[point_idx][3]) /
               40;
        break;

      default: break;
    }

    T abs_error = fabs(pred - s_data[point_idx]);
    atomicAdd(const_cast<T*>(local_errs) + c, abs_error);
  }
  __syncthreads();
  if (TIX < 6) errs[TIX] = local_errs[TIX];
  __syncthreads();
}

template <
    typename T1, typename T2, typename FP, int AnchorBlockSizeX,
    int AnchorBlockSizeY, int AnchorBlockSizeZ,
    int numAnchorBlockX,  // Number of Anchor blocks along X
    int numAnchorBlockY,  // Number of Anchor blocks along Y
    int numAnchorBlockZ,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE, bool WORKFLOW, bool PROBE_PRED_ERROR>
__device__ void cusz::device_api::spline2d_layout2_interpolate(
    volatile T1 s_data[1]
                      [AnchorBlockSizeY * numAnchorBlockY + 1]
                      [AnchorBlockSizeX * numAnchorBlockX + 1],
    volatile T2 s_ectrl[1]
                       [AnchorBlockSizeY * numAnchorBlockY + 1]
                       [AnchorBlockSizeX * numAnchorBlockX + 1],
    DIM3 data_size, FP eb_r, FP ebx2, int radius,
    INTERPOLATION_PARAMS intp_param)
{
  auto xblue = [] __device__(int _tix, int unit) -> int {
    return unit * (_tix * 2);
  };
  auto yblue = [] __device__(int _tiy, int unit) -> int {
    return unit * (_tiy * 2);
  };
  auto zblue = [] __device__(int _tiz, int unit) -> int {
    return unit * (_tiz * 2 + 1);
  };

  auto xblue_reverse = [] __device__(int _tix, int unit) -> int {
    return unit * (_tix);
  };
  auto yblue_reverse = [] __device__(int _tiy, int unit) -> int {
    return unit * (_tiy);
  };
  auto zblue_reverse = [] __device__(int _tiz, int unit) -> int {
    return unit * (_tiz * 2 + 1);
  };

  auto xyellow = [] __device__(int _tix, int unit) -> int {
    return unit * (_tix * 2);
  };
  auto yyellow = [] __device__(int _tiy, int unit) -> int {
    return unit * (_tiy * 2 + 1);
  };
  auto zyellow = [] __device__(int _tiz, int unit) -> int {
    return unit * (_tiz);
  };

  auto xyellow_reverse = [] __device__(int _tix, int unit) -> int {
    return unit * (_tix);
  };
  auto yyellow_reverse = [] __device__(int _tiy, int unit) -> int {
    return unit * (_tiy * 2 + 1);
  };
  auto zyellow_reverse = [] __device__(int _tiz, int unit) -> int {
    return unit * (_tiz * 2);
  };

  auto xhollow = [] __device__(int _tix, int unit) -> int {
    return unit * (_tix * 2 + 1);
  };
  auto yhollow = [] __device__(int _tiy, int unit) -> int {
    return unit * (_tiy);
  };
  auto zhollow = [] __device__(int _tiz, int unit) -> int {
    return unit * (_tiz);
  };

  auto xhollow_reverse = [] __device__(int _tix, int unit) -> int {
    return unit * (_tix * 2 + 1);
  };
  auto yhollow_reverse = [] __device__(int _tiy, int unit) -> int {
    return unit * (_tiy * 2);
  };
  auto zhollow_reverse = [] __device__(int _tiz, int unit) -> int {
    return unit * (_tiz * 2);
  };

  constexpr auto COARSEN = true;
  constexpr auto NO_COARSEN = false;
  constexpr auto BORDER_INCLUSIVE = true;
  constexpr auto BORDER_EXCLUSIVE = false;

  FP cur_ebx2 = ebx2, cur_eb_r = eb_r;

  auto calc_eb = [&](auto unit) {
    cur_ebx2 = ebx2, cur_eb_r = eb_r;
    int temp = 1;
    while (temp < unit) {
      temp *= 2;
      cur_eb_r *= intp_param.alpha;
      cur_ebx2 /= intp_param.alpha;
    }
    if (cur_ebx2 < ebx2 / intp_param.beta) {
      cur_ebx2 = ebx2 / intp_param.beta;
      cur_eb_r = eb_r * intp_param.beta;
    }
  };
  
   if constexpr (AnchorBlockSizeX == 32){
  int unit = 16;
  calc_eb(unit);
  // set_orders(reverse[2]);
  if (intp_param.reverse[2]) {

    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX, numAnchorBlockY + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2 + 1, numAnchorBlockY, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX + 1, numAnchorBlockY, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX, numAnchorBlockY * 2 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }
    unit = 8;
  calc_eb(unit);

  // iteration 2, TODO switch y-z order
  if (intp_param.reverse[1]) {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2, numAnchorBlockY * 2 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4 + 1, numAnchorBlockY * 2, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2 + 1, numAnchorBlockY * 2, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow),
        decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2, numAnchorBlockY * 4 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }
  
  unit = 4;
  calc_eb(unit);

  // iteration 3
  if (intp_param.reverse[0]) {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4, numAnchorBlockY * 4 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 8 + 1, numAnchorBlockY * 4, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);

    // may have bug end
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4 + 1, numAnchorBlockY * 4, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow),
        decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4, numAnchorBlockY * 8 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }

  unit = 2;
  calc_eb(unit);

  // iteration 3
  if (intp_param.reverse[0]) {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 8, numAnchorBlockY * 8 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 16 + 1, numAnchorBlockY * 8, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);

    // may have bug end
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 8 + 1, numAnchorBlockY * 8, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow),
        decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 8, numAnchorBlockY * 16 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }

  unit = 1;
  calc_eb(unit);

  // iteration 3
  if (intp_param.reverse[0]) {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 16, numAnchorBlockY * 16 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 32 + 1, numAnchorBlockY * 16, NO_COARSEN, 1,
        BORDER_EXCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);

    // may have bug end
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 16 + 1, numAnchorBlockY * 16, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow),
        decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 16, numAnchorBlockY * 32 + 1, NO_COARSEN, 1,
        BORDER_EXCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }
   }

   if constexpr (AnchorBlockSizeX == 16){
  int unit = 8;
  calc_eb(unit);
  // set_orders(reverse[2]);
  if (intp_param.reverse[2]) {

    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX, numAnchorBlockY + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2 + 1, numAnchorBlockY, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2, numAnchorBlockY, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX, numAnchorBlockY * 2 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }
  unit = 4;
  calc_eb(unit);

  // iteration 2, TODO switch y-z order
  if (intp_param.reverse[1]) {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2, numAnchorBlockY * 2 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4 + 1, numAnchorBlockY * 2, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2 + 1, numAnchorBlockY * 2, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow),
        decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2, numAnchorBlockY * 4 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }
  
  unit = 2;
  calc_eb(unit);

  // iteration 3
  if (intp_param.reverse[0]) {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4, numAnchorBlockY * 4 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 8 + 1, numAnchorBlockY * 4, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);

    // may have bug end
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4 + 1, numAnchorBlockY * 4, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow),
        decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4, numAnchorBlockY * 8 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }

  unit = 1;
  calc_eb(unit);

  // iteration 3
  if (intp_param.reverse[0]) {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 8, numAnchorBlockY * 8 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 16 + 1, numAnchorBlockY * 8, NO_COARSEN, 1,
        BORDER_EXCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);

    // may have bug end
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 8 + 1, numAnchorBlockY * 8, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow),
        decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 8, numAnchorBlockY * 16 + 1, NO_COARSEN, 1,
        BORDER_EXCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }
  }

 if constexpr (AnchorBlockSizeX == 8){
  int unit = 4;
  calc_eb(unit);
  // set_orders(reverse[2]);
  if (intp_param.reverse[2]) {

    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX, numAnchorBlockY + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2 + 1, numAnchorBlockY, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2, numAnchorBlockY, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX, numAnchorBlockY * 2 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }
  unit = 2;
  calc_eb(unit);

  // iteration 2, TODO switch y-z order
  if (intp_param.reverse[1]) {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2, numAnchorBlockY * 2 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4 + 1, numAnchorBlockY * 2, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2 + 1, numAnchorBlockY * 2, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow),
        decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 2, numAnchorBlockY * 4 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }
  
  unit = 1;
  calc_eb(unit);

  // iteration 3
  if (intp_param.reverse[0]) {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4, numAnchorBlockY * 4 + 1, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow_reverse, yhollow_reverse,
        zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[1]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 8 + 1, numAnchorBlockY * 4, NO_COARSEN, 1,
        BORDER_EXCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow_reverse, yyellow_reverse,
        zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius,
        intp_param.interpolators[2]);

    // may have bug end
  }
  else {
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4 + 1, numAnchorBlockY * 4, NO_COARSEN, 1,
        BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[0]);
    interpolate_stage<
        T1, T2, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        decltype(xhollow), decltype(yhollow),
        decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, numAnchorBlockX * 4, numAnchorBlockY * 8 + 1, NO_COARSEN, 1,
        BORDER_EXCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r,
        cur_ebx2, radius, intp_param.interpolators[1]);
  }
 }
}

/********************************************************************************
 * host API/kernel
 ********************************************************************************/
template <typename TITER, int LINEAR_BLOCK_SIZE>
__global__ void cusz::c_spline2d_profiling_16x16x16data(
    TITER data, DIM3 data_size, STRIDE3 data_leap, TITER errors)
{
  // compile time variables
  using T = typename std::remove_pointer<TITER>::type;

  {
    __shared__ struct {
      T data[16][16][16];
      T local_errs[2];
      // T global_errs[6];
    } shmem;

    c_reset_scratch_profiling_16x16x16data<T, LINEAR_BLOCK_SIZE>(
        shmem.data, 0.0);
    global2shmem_profiling_16x16x16data<T, T, LINEAR_BLOCK_SIZE>(
        data, data_size, data_leap, shmem.data);

    cusz::device_api::auto_tuning_2d<T, LINEAR_BLOCK_SIZE>(
        shmem.data, shmem.local_errs, data_size, errors);
  }
}

template <typename TITER, int LINEAR_BLOCK_SIZE>
__global__ void cusz::c_spline2d_profiling_data_2(
    TITER data, DIM3 data_size, STRIDE3 data_leap, TITER errors)
{
  // compile time variables
  using T = typename std::remove_pointer<TITER>::type;

  {
    __shared__ struct {
      T data[64];
      T neighbor_x[64][4];
      T neighbor_y[64][4];
      T neighbor_z[64][4];
      T local_errs[6];
      // T global_errs[6];
    } shmem;

    c_reset_scratch_profiling_data_2<T, LINEAR_BLOCK_SIZE>(
        shmem.data, shmem.neighbor_x, shmem.neighbor_y, shmem.neighbor_z, 0.0);
    global2shmem_profiling_data_2<T, T, LINEAR_BLOCK_SIZE>(
        data, data_size, data_leap, shmem.data, shmem.neighbor_x,
        shmem.neighbor_y, shmem.neighbor_z);

    if (TIX < 6 and BIX == 0 and BIY == 0 and BIZ == 0)
      errors[TIX] = 0.0;  // risky
    cusz::device_api::auto_tuning_2_2d<T, LINEAR_BLOCK_SIZE>(
        shmem.data, shmem.neighbor_x, shmem.neighbor_y, shmem.neighbor_z,
        shmem.local_errs, data_size, errors);
  }
}

template <
    typename TITER, typename EITER, typename FP, int AnchorBlockSizeX,
    int AnchorBlockSizeY, int AnchorBlockSizeZ,
    int numAnchorBlockX,  // Number of Anchor blocks along X
    int numAnchorBlockY,  // Number of Anchor blocks along Y
    int numAnchorBlockZ,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE, typename CompactVal, typename CompactIdx,
    typename CompactNum>
__global__ void cusz::c_spline2d_infprecis_data(
    TITER data, DIM3 data_size, STRIDE3 data_leap, EITER ectrl,
    DIM3 ectrl_size, STRIDE3 ectrl_leap, TITER anchor, STRIDE3 anchor_leap,
    CompactVal compact_val, CompactIdx compact_idx, CompactNum compact_num,
    FP eb_r, FP ebx2, int radius,
    INTERPOLATION_PARAMS intp_param  //,
                                     // TITER errors
)
{
  // compile time variables
  using T = typename std::remove_pointer<TITER>::type;
  using E = typename std::remove_pointer<EITER>::type;

  {
    __shared__ struct {
      T data[1]
            [AnchorBlockSizeY * numAnchorBlockY + 1]
            [AnchorBlockSizeX * numAnchorBlockX + 1];
      T ectrl[1]
             [AnchorBlockSizeY * numAnchorBlockY + 1]
             [AnchorBlockSizeX * numAnchorBlockX + 1];

      // T global_errs[6];
    } shmem;
    // if(TIX + TIY + TIZ == 0 && BIX + BIY + BIZ == 0) printf("blockdim=%d %d %d, gridDim=%d %d %d\n", BDX, BDY, BDZ, GDX, GDY, GDZ);
    c_reset_scratch_data<
        T, T, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        LINEAR_BLOCK_SIZE>(shmem.data, shmem.ectrl, radius);

    global2shmem_data<
        T, T, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        LINEAR_BLOCK_SIZE>(data, data_size, data_leap, shmem.data);
    c_gather_anchor<
        T, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ  // Number of Anchor blocks along Z
        >(data, data_size, data_leap, anchor, anchor_leap);

    cusz::device_api::spline2d_layout2_interpolate<
        T, T, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        LINEAR_BLOCK_SIZE, SPLINE3_COMPR, false>(
        shmem.data, shmem.ectrl, data_size, eb_r, ebx2, radius, intp_param);

    shmem2global_data_with_compaction<
        T, E, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX,  // Number of Anchor blocks along X
        numAnchorBlockY,  // Number of Anchor blocks along Y
        numAnchorBlockZ,  // Number of Anchor blocks along Z
        LINEAR_BLOCK_SIZE>(
        shmem.ectrl, ectrl, ectrl_size, ectrl_leap, radius, compact_val,
        compact_idx, compact_num);
  }
}

template <
    typename EITER, typename TITER, typename FP, int AnchorBlockSizeX,
    int AnchorBlockSizeY, int AnchorBlockSizeZ,
    int numAnchorBlockX,  // Number of Anchor blocks along X
    int numAnchorBlockY,  // Number of Anchor blocks along Y
    int numAnchorBlockZ,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE>
__global__ void cusz::x_spline2d_infprecis_data(
    EITER ectrl,          // input 1
    DIM3 ectrl_size,      //
    STRIDE3 ectrl_leap,   //
    TITER anchor,         // input 2
    DIM3 anchor_size,     //
    STRIDE3 anchor_leap,  //
    TITER data,           // output
    DIM3 data_size,       //
    STRIDE3 data_leap,    //
    FP eb_r, FP ebx2, int radius, INTERPOLATION_PARAMS intp_param)
{
  // compile time variables
  using E = typename std::remove_pointer<EITER>::type;
  using T = typename std::remove_pointer<TITER>::type;

  __shared__ struct {
    T data[1]
          [AnchorBlockSizeY * numAnchorBlockY + 1]
          [AnchorBlockSizeX * numAnchorBlockX + 1];
    T ectrl[1]
           [AnchorBlockSizeY * numAnchorBlockY + 1]
           [AnchorBlockSizeX * numAnchorBlockX + 1];
  } shmem;
  // if(TIX + TIY + TIZ == 0 && BIX + BIY + BIZ == 0) printf("blockdim=%d %d %d, gridDim=%d %d %d\n", BDX, BDY, BDZ, GDX, GDY, GDZ);
  x_reset_scratch_data<
      T, T, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
      numAnchorBlockX,  // Number of Anchor blocks along X
      numAnchorBlockY,  // Number of Anchor blocks along Y
      numAnchorBlockZ,  // Number of Anchor blocks along Z
      LINEAR_BLOCK_SIZE>(
      shmem.data, shmem.ectrl, anchor, anchor_size, anchor_leap);
  global2shmem_fuse<
      T, E, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
      numAnchorBlockX,  // Number of Anchor blocks along X
      numAnchorBlockY,  // Number of Anchor blocks along Y
      numAnchorBlockZ,  // Number of Anchor blocks along Z
      LINEAR_BLOCK_SIZE>(ectrl, ectrl_size, ectrl_leap, data, shmem.ectrl);

  cusz::device_api::spline2d_layout2_interpolate<
      T, T, FP, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
      numAnchorBlockX,  // Number of Anchor blocks along X
      numAnchorBlockY,  // Number of Anchor blocks along Y
      numAnchorBlockZ,  // Number of Anchor blocks along Z
      LINEAR_BLOCK_SIZE, SPLINE3_DECOMPR, false>(
      shmem.data, shmem.ectrl, data_size, eb_r, ebx2, radius, intp_param);
  shmem2global_data<
      T, T, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
      numAnchorBlockX,  // Number of Anchor blocks along X
      numAnchorBlockY,  // Number of Anchor blocks along Y
      numAnchorBlockZ,  // Number of Anchor blocks along Z
      LINEAR_BLOCK_SIZE>(shmem.data, data, data_size, data_leap);
}

#undef TIX
#undef TIY
#undef TIZ
#undef BIX
#undef BIY
#undef BIZ
#undef BDX
#undef BDY
#undef BDZ
#undef GDX
#undef GDY
#undef GDZ

#endif