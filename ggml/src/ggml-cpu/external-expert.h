#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cpu_external_expert_lease {
    const void * data;
    void * token;
    uint64_t generation;
};

struct ggml_cpu_external_expert_prefetch {
    struct ggml_cpu_external_expert_lease (*acquire)(void *, int64_t, size_t, const void *);
    void (*issue)(void *, const void *, int64_t, size_t);
    uint64_t (*compute_begin)(void *);
    void (*compute_end)(void *, const void *, uint64_t);
    void (*finish)(void *, const void *);
};

// A null data pointer means failure. The provider rolls back any reservation before returning it.
struct ggml_cpu_external_expert_source {
    void * context;
    struct ggml_cpu_external_expert_lease (*acquire)(void * context, int64_t expert, size_t bytes);
    void (*release)(void * context, struct ggml_cpu_external_expert_lease lease);
    const struct ggml_cpu_external_expert_prefetch * prefetch
#ifdef __cplusplus
        = nullptr
#endif
        ;
};

GGML_BACKEND_API ggml_backend_buffer_t ggml_cpu_external_expert_buffer_create(void);

#ifdef __cplusplus
}
#endif
