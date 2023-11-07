/**
 * @file capsule.h
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-09
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef C00A17B2_29D2_47F0_B667_E6814586B4EB
#define C00A17B2_29D2_47F0_B667_E6814586B4EB

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "cusz/type.h"

typedef struct psz_memory_segment {
  char name[10];
  psz_dtype type;
  int tsize;
  void* buf;  // compat for wip
  void *d, *h, *uni;
  size_t len{1}, bytes{1};
  uint32_t lx{1}, ly{1}, lz{1};
  size_t sty{1}, stz{1};  // stride
  bool isaview{false}, d_borrowed{false}, h_borrowed{false};

} pszmem;

void pszmem__calc_len(pszmem* m);
int pszmem__ndim(pszmem* m);
void pszmem__dbg(pszmem* m);
pszmem* pszmem_create1(psz_dtype t, u4 lx);
pszmem* pszmem_create2(psz_dtype t, u4 lx, u4 ly);
pszmem* pszmem_create3(psz_dtype t, u4 lx, u4 ly, u4 lz);
void pszmem_borrow(pszmem* m, void* _d, void* _h);
void pszmem_setname(pszmem* m, const char name[10]);
void pszmem_clearhost(pszmem* m);
void pszmem_fromfile(const char* fname, pszmem* m);
void pszmem_tofile(const char* fname, pszmem* m);
void pszmem_viewas(pszmem* backend, pszmem* frontend);

void pszmem_malloc_cuda(pszmem* m);
void pszmem_mallochost_cuda(pszmem* m);
void pszmem_mallocmanaged_cuda(pszmem* m);
void pszmem_free_cuda(pszmem* m);
void pszmem_freehost_cuda(pszmem* m);
void pszmem_freemanaged_cuda(pszmem* m);
void pszmem_cleardevice_cuda(pszmem* m);
void pszmem_h2d_cuda(pszmem* m);
void pszmem_h2d_cudaasync(pszmem* m, void* stream);
void pszmem_d2h_cuda(pszmem* m);
void pszmem_d2h_cudaasync(pszmem* m, void* stream);
void pszmem_device_deepcopy_cuda(pszmem* dst, pszmem* src);
void pszmem_host_deepcopy_cuda(pszmem* dst, pszmem* src);

void pszmem_malloc_hip(pszmem* m);
void pszmem_mallochost_hip(pszmem* m);
void pszmem_mallocmanaged_hip(pszmem* m);
void pszmem_free_hip(pszmem* m);
void pszmem_freehost_hip(pszmem* m);
void pszmem_freemanaged_hip(pszmem* m);
void pszmem_cleardevice_hip(pszmem* m);
void pszmem_h2d_hip(pszmem* m);
void pszmem_h2d_hipasync(pszmem* m, void* stream);
void pszmem_d2h_hip(pszmem* m);
void pszmem_d2h_hipasync(pszmem* m, void* stream);
void pszmem_device_deepcopy_hip(pszmem* dst, pszmem* src);
void pszmem_host_deepcopy_hip(pszmem* dst, pszmem* src);

void pszmem_malloc_1api(pszmem* m);
void pszmem_mallochost_1api(pszmem* m);
void pszmem_mallocshared_1api(pszmem* m);
void pszmem_free_1api(pszmem* m);
void pszmem_freehost_1api(pszmem* m);
void pszmem_freeshared_1api(pszmem* m);
void pszmem_cleardevice_1api(pszmem* m);
void pszmem_h2d_1api(pszmem* m);
void pszmem_h2d_1apiasync(pszmem* m, void* stream);
void pszmem_d2h_1api(pszmem* m);
void pszmem_d2h_1apiasync(pszmem* m, void* stream);
void pszmem_device_deepcopy_1api(pszmem* dst, pszmem* src);
void pszmem_host_deepcopy_1api(pszmem* dst, pszmem* src);

void pszmem_malloc_cuda(pszmem* m)
{
  if (m->d_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed dptr.");

  if (m->d == nullptr) {
    if (not m->isaview)
      CHECK_GPU(cudaMalloc(&m->d, m->bytes));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": dptr already malloc'ed.");
  }
  CHECK_GPU(cudaMemset(m->d, 0x0, m->bytes));
}

void pszmem_mallochost_cuda(pszmem* m)
{
  if (m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed hptr.");

  if (m->h == nullptr) {
    if (not m->isaview)
      CHECK_GPU(cudaMallocHost(&m->h, m->bytes));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": hptr already malloc'ed.");
  }
  memset(m->h, 0x0, m->bytes);
}

void pszmem_cleardevice_cuda(pszmem* m)
{
  CHECK_GPU(cudaMemset(m->d, 0x0, m->bytes));
}

void pszmem_mallocmanaged_cuda(pszmem* m)
{
  if (m->d_borrowed or m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed uniptr.");

  if (m->uni == nullptr) {
    if (not m->isaview)
      CHECK_GPU(cudaMallocManaged(&m->uni, m->bytes));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": uniptr already malloc'ed.");
  }
  CHECK_GPU(cudaMemset(m->uni, 0x0, m->bytes));
}

void pszmem_free_cuda(pszmem* m)
{
  if (m->d_borrowed)
    throw std::runtime_error(string(m->name) + ": cannot free borrowed dptr");

  if (m->d) {
    if (not m->isaview)
      CHECK_GPU(cudaFree(m->d));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_freehost_cuda(pszmem* m)
{
  if (m->h_borrowed)
    throw std::runtime_error(string(m->name) + ": cannot free borrowed hptr.");

  if (m->h) {
    if (not m->isaview)
      CHECK_GPU(cudaFreeHost(m->h));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_freemanaged_cuda(pszmem* m)
{
  if (m->d_borrowed or m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot free borrowed uniptr.");

  if (m->uni) {
    if (not m->isaview)
      CHECK_GPU(cudaFree(m->uni));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_h2d_cuda(pszmem* m)
{
  CHECK_GPU(cudaMemcpy(m->d, m->h, m->bytes, cudaMemcpyHostToDevice));
}

void pszmem_h2d_cudaasync(pszmem* m, void* stream)
{
  CHECK_GPU(cudaMemcpyAsync(
      m->d, m->h, m->bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream));
}

void pszmem_d2h_cuda(pszmem* m)
{
  CHECK_GPU(cudaMemcpy(m->h, m->d, m->bytes, cudaMemcpyDeviceToHost));
}

void pszmem_d2h_cudaasync(pszmem* m, void* stream)
{
  CHECK_GPU(cudaMemcpyAsync(
      m->h, m->d, m->bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
}

void pszmem_device_deepcopy_cuda(pszmem* dst, pszmem* src)
{
  CHECK_GPU(cudaMemcpy(dst->d, src->d, src->bytes, cudaMemcpyDeviceToDevice));
}

void pszmem_host_deepcopy_cuda(pszmem* dst, pszmem* src)
{
  CHECK_GPU(cudaMemcpy(dst->h, src->h, src->bytes, cudaMemcpyHostToHost));
}

// no impl. in C due to typing issue
// void pszmem_cast(pszmem* dst, pszmem* src, psz_space s);

#ifdef __cplusplus
}
#endif

#endif /* C00A17B2_29D2_47F0_B667_E6814586B4EB */
