#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>

// Platform headers for mmap
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace qwen3_tts {

// ============================================================
// Thread-safe shared backend singleton
// ============================================================
// All components (TTSTransformer, AudioTokenizerDecoder, etc.) share
// a single GPU backend to avoid creating multiple Metal/CUDA contexts.
// A mutex protects init/release against concurrent qwen3_tts_create() calls.
namespace {
struct shared_backend_state {
    ggml_backend_t backend   = nullptr;
    int32_t        ref_count = 0;
    std::mutex     mtx;
};

shared_backend_state & get_shared_backend_state() {
    static shared_backend_state state;
    return state;
}
} // anonymous namespace

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() {
    close();
}

ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg) {
    if (error_msg) error_msg->clear();

    // Debug escape hatch: force the CPU backend even when a GPU is present.
    const char * force_cpu = getenv("QWEN3_TTS_FORCE_CPU");

    // Per-component backend override (e.g. QWEN3_TTS_TRANSFORMER_BACKEND=cpu)
    bool force_component_cpu = (force_cpu && force_cpu[0] == '1');
    if (component_name) {
        if (strcmp(component_name, "TTSTransformer") == 0) {
            const char * env = getenv("QWEN3_TTS_TRANSFORMER_BACKEND");
            if (env && (strcmp(env, "cpu") == 0 || strcmp(env, "CPU") == 0)) {
                force_component_cpu = true;
            }
        } else if (strcmp(component_name, "AudioTokenizerDecoder") == 0) {
            const char * env = getenv("QWEN3_TTS_DECODER_BACKEND");
            if (env && (strcmp(env, "cpu") == 0 || strcmp(env, "CPU") == 0)) {
                force_component_cpu = true;
            }
        }
    }

    if (force_component_cpu) {
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    auto & shared = get_shared_backend_state();
    std::lock_guard<std::mutex> lock(shared.mtx);

    if (shared.backend) {
        shared.ref_count++;
        return shared.backend;
    }

    ggml_backend_t backend = nullptr;
    backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    if (!backend && error_msg) {
        const char * name = component_name ? component_name : "component";
        *error_msg = "Failed to initialize backend (IGPU/GPU/ACCEL/CPU) for " + std::string(name);
    }

    if (backend) {
        shared.backend   = backend;
        shared.ref_count = 1;
    }

    return backend;
}

void release_preferred_backend(ggml_backend_t backend) {
    if (!backend) return;

    auto & shared = get_shared_backend_state();
    std::lock_guard<std::mutex> lock(shared.mtx);

    if (shared.backend == backend) {
        shared.ref_count--;
        if (shared.ref_count <= 0) {
            ggml_backend_free(shared.backend);
            shared.backend   = nullptr;
            shared.ref_count = 0;
        }
        return;
    }

    // Backend not in the shared pool — free directly.
    ggml_backend_free(backend);
}

bool GGUFLoader::open(const std::string & path) {
    close();

    file_path_ = path;

    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx_,
    };

    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) {
        error_msg_ = "Failed to open GGUF file: " + path;
        return false;
    }

    return true;
}

void GGUFLoader::close() {
    if (ctx_) {
        gguf_free(ctx_);
        ctx_ = nullptr;
    }
    if (meta_ctx_) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
    file_path_.clear();
}

int64_t GGUFLoader::get_n_tensors() const {
    if (!ctx_) return 0;
    return gguf_get_n_tensors(ctx_);
}

const char * GGUFLoader::get_tensor_name(int64_t idx) const {
    if (!ctx_) return nullptr;
    return gguf_get_tensor_name(ctx_, idx);
}

enum ggml_type GGUFLoader::get_tensor_type(int64_t idx) const {
    if (!ctx_) return GGML_TYPE_F32;
    return gguf_get_tensor_type(ctx_, idx);
}

size_t GGUFLoader::get_tensor_offset(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_offset(ctx_, idx);
}

size_t GGUFLoader::get_tensor_size(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_size(ctx_, idx);
}

int32_t GGUFLoader::get_u32(const char * key, int32_t default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return (int32_t)gguf_get_val_u32(ctx_, idx);
}

float GGUFLoader::get_f32(const char * key, float default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return gguf_get_val_f32(ctx_, idx);
}

size_t GGUFLoader::get_data_offset() const {
    if (!ctx_) return 0;
    return gguf_get_data_offset(ctx_);
}

// ============================================================
// mmap helpers (RAII)
// ============================================================
namespace {

struct MmapFile {
    const uint8_t * data = nullptr;
    size_t          size = 0;

#ifdef _WIN32
    HANDLE hFile   = INVALID_HANDLE_VALUE;
    HANDLE hMap    = nullptr;
#else
    int    fd      = -1;
#endif

    bool open(const std::string & path) {
#ifdef _WIN32
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER li;
        if (!GetFileSizeEx(hFile, &li)) { close(); return false; }
        size = (size_t)li.QuadPart;

        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) { close(); return false; }

        data = (const uint8_t *)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!data) { close(); return false; }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;

        struct stat st;
        if (fstat(fd, &st) != 0) { close(); return false; }
        size = (size_t)st.st_size;

        void * ptr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) { close(); return false; }

        // Advise the kernel this will be read sequentially (prefetches pages)
        madvise(ptr, size, MADV_SEQUENTIAL);
        data = (const uint8_t *)ptr;
#endif
        return true;
    }

    void close() {
#ifdef _WIN32
        if (data)  { UnmapViewOfFile(data); data = nullptr; }
        if (hMap)  { CloseHandle(hMap);     hMap = nullptr; }
        if (hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile);
            hFile = INVALID_HANDLE_VALUE;
        }
#else
        if (data)  { munmap((void *)data, size); data = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
#endif
        size = 0;
    }

    ~MmapFile() { close(); }

    // Non-copyable
    MmapFile() = default;
    MmapFile(const MmapFile &) = delete;
    MmapFile & operator=(const MmapFile &) = delete;
};

} // anonymous namespace

// ============================================================
// load_tensor_data_from_file
// ============================================================
// Uses mmap for zero-copy loading:
//   - CPU backend: tensor data is served directly from the mmap — the OS
//     pages it in on demand and can share pages across processes.
//   - GPU backends (Metal/CUDA/Vulkan): mmap avoids a per-tensor heap
//     allocation; we pass the mmap pointer directly to ggml_backend_tensor_set
//     which copies it to VRAM in a single call per tensor.
//
// Falls back to fread if mmap is unavailable (e.g. network filesystem).
bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    enum ggml_backend_dev_type preferred_backend_type,
    ggml_backend_t backend_in
) {
    const char * force_cpu = getenv("QWEN3_TTS_FORCE_CPU");
    ggml_backend_t backend = nullptr;
    bool using_shared = false;
    if (backend_in) {
        backend = backend_in;
        using_shared = false;
    } else if ((force_cpu && force_cpu[0] == '1') || preferred_backend_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    } else {
        backend = init_preferred_backend("GGUFLoader", &error_msg);
        using_shared = (backend != nullptr);
    }
    if (!backend) {
        error_msg = "Failed to initialize backend for GGUF tensor loader";
        return false;
    }
    auto release_backend = [&]() {
        if (backend_in) {
            // Caller owns backend_in — do not release
        } else if (using_shared) {
            release_preferred_backend(backend);
        } else {
            ggml_backend_free(backend);
        }
    };

    buffer = ggml_backend_alloc_ctx_tensors(model_ctx, backend);
    if (!buffer) {
        error_msg = "Failed to allocate tensor buffer";
        release_backend();
        return false;
    }

    const size_t  data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors   = gguf_get_n_tensors(ctx);

    // ---- Try mmap first -------------------------------------------------
    MmapFile mf;
    if (mf.open(path) && mf.data && mf.size > data_offset) {
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name(ctx, i);
            size_t       offset = gguf_get_tensor_offset(ctx, i);

            auto it = tensors.find(name);
            if (it == tensors.end()) continue;

            struct ggml_tensor * tensor = it->second;
            size_t nbytes = ggml_nbytes(tensor);

            if (data_offset + offset + nbytes > mf.size) {
                error_msg = "mmap: tensor data out of bounds: " + std::string(name);
                release_backend();
                return false;
            }

            // Point directly into the mmap — no intermediate copy on CPU
            ggml_backend_tensor_set(tensor, mf.data + data_offset + offset, 0, nbytes);
        }
        release_backend();
        return true;
    }

    // ---- mmap unavailable — fall back to fread --------------------------
    mf.close();
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg = "Failed to open file for reading: " + path;
        release_backend();
        return false;
    }

    std::vector<uint8_t> read_buf;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name   = gguf_get_tensor_name(ctx, i);
        size_t       offset = gguf_get_tensor_offset(ctx, i);

        auto it = tensors.find(name);
        if (it == tensors.end()) continue;

        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);

        read_buf.resize(nbytes);

        if (
#ifdef _WIN32
            _fseeki64(f, (int64_t)(data_offset + offset), SEEK_SET) != 0
#else
            fseeko(f, (off_t)(data_offset + offset), SEEK_SET) != 0
#endif
        ) {
            error_msg = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            release_backend();
            return false;
        }

        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            release_backend();
            return false;
        }

        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }

    fclose(f);
    release_backend();
    return true;
}

void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer) {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx)    ggml_free(ctx);
}

} // namespace qwen3_tts
