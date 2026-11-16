#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cpu_external_expert_lease {
    const void * data;
    void * token;
};

// A null data pointer means failure. The provider rolls back any reservation before returning it.
struct ggml_cpu_external_expert_source {
    void * context;
    struct ggml_cpu_external_expert_lease (*acquire)(void * context, int64_t expert, size_t bytes);
    void (*release)(void * context, void * token);
};

GGML_BACKEND_API ggml_backend_buffer_t ggml_cpu_external_expert_buffer_create(void);

#ifdef __cplusplus
}
#endif
