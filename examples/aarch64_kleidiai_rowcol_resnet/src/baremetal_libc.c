#include <stddef.h>
#include <stdint.h>

// Bare metal implementations of standard library functions needed by xxHash

void* memcpy(void* dest, const void* src, size_t n) {
    char* d = (char*)dest;
    const char* s = (const char*)src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
    return dest;
}

void* memset(void* s, int c, size_t n) {
    char* p = (char*)s;
    for (size_t i = 0; i < n; i++) {
        p[i] = (char)c;
    }
    return s;
}

size_t strlen(const char* s) {
    size_t len = 0;
    while (s[len] != '\0') {
        len++;
    }
    return len;
}

int memcmp(const void* s1, const void* s2, size_t n) {
    const unsigned char* p1 = (const unsigned char*)s1;
    const unsigned char* p2 = (const unsigned char*)s2;
    for (size_t i = 0; i < n; i++) {
        if (p1[i] < p2[i]) {
            return -1;
        } else if (p1[i] > p2[i]) {
            return 1;
        }
    }
    return 0;
}

// Stub implementations for malloc/free (should not be used with proper xxHash configuration)
void* malloc(size_t size) {
    // In bare metal environment, we don't want to use malloc
    // This should not be called if xxHash is configured properly
    (void)size;
    return NULL;
}

void free(void* ptr) {
    // In bare metal environment, we don't want to use free
    // This should not be called if xxHash is configured properly
    (void)ptr;
}

void __assert_func(const char* file, int line, const char* func, const char* expr) {
    // In bare metal, we can't print, so just hang
    while (1) {
        // Infinite loop on assertion failure
        __asm__ volatile("wfi"); // Wait for interrupt to save power
    }
}
