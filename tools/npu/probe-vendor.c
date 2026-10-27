// Vendor-loading approach follows the public S24 Ultra sphal integration.
// FastRPC ABI constants/structures follow Qualcomm's BSD-3-Clause inc/remote.h.
#include <android/dlext.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct android_namespace_t *(*get_namespace_fn)(const char *);
typedef void *(*open_ext_fn)(const char *, int, const android_dlextinfo *);
typedef int (*control_fn)(uint32_t, void *, uint32_t);
typedef void *(*alloc_fn)(int, uint32_t, int);
typedef void (*free_fn)(void *);
typedef int (*to_fd_fn)(void *);
struct capability {
  uint32_t domain;
  uint32_t attribute_id;
  uint32_t value;
};

int main(void) {
  get_namespace_fn get_namespace = (get_namespace_fn)dlsym(
      RTLD_DEFAULT, "__loader_android_get_exported_namespace");
  open_ext_fn open_ext = (open_ext_fn)dlsym(RTLD_DEFAULT, "android_dlopen_ext");
  struct android_namespace_t *ns =
      get_namespace ? get_namespace("sphal") : NULL;
  printf("RESULT {\"phase\":\"namespace\",\"api\":%s,\"sphal\":%s}\n",
         get_namespace && open_ext ? "true" : "false", ns ? "true" : "false");
  if (!ns || !open_ext)
    return 2;
  android_dlextinfo info = {0};
  info.flags = ANDROID_DLEXT_USE_NAMESPACE;
  info.library_namespace = ns;
  void *driver =
      open_ext("/vendor/lib64/libcdsprpc.so", RTLD_NOW | RTLD_LOCAL, &info);
  if (!driver) {
    printf("ERROR driver_load: %s\n", dlerror());
    return 3;
  }
  const char *symbols[] = {"rpcmem_alloc",
                           "rpcmem_alloc2",
                           "rpcmem_free",
                           "rpcmem_to_fd",
                           "fastrpc_mmap",
                           "fastrpc_munmap",
                           "dspqueue_create",
                           "dspqueue_close",
                           "dspqueue_export",
                           "dspqueue_write",
                           "dspqueue_read",
                           "remote_handle64_open",
                           "remote_handle64_invoke",
                           "remote_handle_control",
                           "remote_handle64_control",
                           "remote_session_control",
                           "remote_handle64_close"};
  int missing = 0;
  for (unsigned i = 0; i < sizeof(symbols) / sizeof(symbols[0]); ++i) {
    int present = dlsym(driver, symbols[i]) != NULL;
    printf("RESULT {\"phase\":\"symbol\",\"name\":\"%s\",\"present\":%s}\n",
           symbols[i], present ? "true" : "false");
    if (!present && strcmp(symbols[i], "rpcmem_alloc2"))
      ++missing;
  }
  if (missing)
    return 4;
  control_fn control = (control_fn)dlsym(driver, "remote_handle_control");
  const char *attributes[] = {
      "domain_support", "unsigned_pd_support", "hvx64",
      "hvx128",         "vtcm_page",           "vtcm_count",
      "architecture",   "hmx_depth",           "hmx_spatial"};
  int arch_status = -1;
  uint32_t arch_value = 0;
  for (uint32_t i = 0; i < 9; ++i) {
    struct capability cap = {
        3, i, 0}; // CDSP domain; public remote_dsp_attributes order.
    int status = control(2, &cap, sizeof(cap)); // DSPRPC_GET_DSP_INFO.
    printf("RESULT "
           "{\"phase\":\"capability\",\"attribute\":\"%s\",\"status\":%d,"
           "\"value\":%u}\n",
           attributes[i], status, cap.value);
    if (i == 6) {
      arch_status = status;
      arch_value = cap.value;
    }
  }
  alloc_fn alloc = (alloc_fn)dlsym(driver, "rpcmem_alloc");
  free_fn release = (free_fn)dlsym(driver, "rpcmem_free");
  to_fd_fn to_fd = (to_fd_fn)dlsym(driver, "rpcmem_to_fd");
  void *memory =
      alloc(25, 1, 4096); // Non-secure system heap, cached allocation.
  int fd = memory ? to_fd(memory) : -1;
  if (memory) {
    memset(memory, 0, 4096);
    release(memory);
  }
  printf("RESULT {\"phase\":\"rpcmem\",\"allocated\":%s,\"valid_fd\":%s}\n",
         memory ? "true" : "false", fd >= 0 ? "true" : "false");
  puts("NOTE capability queries and shared-memory allocation are not DSP "
       "numerical execution.");
  // Keep the vendor runtime loaded until process exit; it owns its listeners.
  return arch_status || (arch_value & 255) != 0x79 || fd < 0 ? 5 : 0;
}
