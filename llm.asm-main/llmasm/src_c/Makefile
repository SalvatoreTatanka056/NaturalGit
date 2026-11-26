# Makefile for compiling libllmasm shared library (C inference engine)

CC ?= gcc
CFLAGS ?= -O3 -fPIC -Wall -Wextra
SIMD_FLAGS ?= -mavx2 -mfma -mf16c

# Target name based on OS
ifeq ($(OS),Windows_NT)
    TARGET = llmasm.dll
    LDFLAGS = -shared -Wl,--out-implib,libllmasm.a
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        TARGET = libllmasm.dylib
        LDFLAGS = -shared -fPIC
    else
        TARGET = libllmasm.so
        LDFLAGS = -shared -fPIC -lm
    endif
endif

all: $(TARGET)

$(TARGET): llmasm_engine.c llmasm_engine.h
	$(CC) $(CFLAGS) $(SIMD_FLAGS) $(LDFLAGS) -o $@ llmasm_engine.c

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o *.a *.dylib *.so *.dll

.PHONY: all clean
