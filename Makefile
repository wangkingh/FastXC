# ----------------------------------------------------
#  fastxc  —  顶层 Makefile
#  • make            → Release 并行（默认）
#  • make MODE=seq   → Release 串行
#  • make debug      → Debug  并行
#  • make debug MODE=seq → Debug  串行
# ----------------------------------------------------

# ---------- 编译器 ----------
CC        ?= gcc
NVCC      ?= /usr/local/cuda/bin/nvcc
ARCH      ?= sm_89

# ---------- Release / Debug ----------
CFLAGS        ?= -O3 -Wall
NVCCFLAGS     ?= -O3 --use_fast_math --generate-line-info -arch=$(ARCH)

DBG_CFLAGS    := -O0 -g -Wall
DBG_NVCCFLAGS := -O0 -g -G -lineinfo -arch=$(ARCH)

# ---------- 向子 make 继承 ----------
export CC NVCC ARCH
export CFLAGS NVCCFLAGS

# ---------- 并行 / 串行 开关 ----------
MODE ?= par            # par (默认) | seq
ifeq ($(MODE),par)
  PARFLAG = -j         # 并行
else
  PARFLAG =            # 串行
endif

# ---------- GNU make ≥4 输出同步 ----------
ifeq ($(shell expr $(firstword $(subst ., ,$(MAKE_VERSION))) \>= 4),1)
  OSYNC ?= --output-sync=target
else
  OSYNC :=
endif

# ---------- 子目录 ----------
SUBDIRS := \
    src/sac2spec          src/sac2spec_butter \
    src/sac2spec_cos      src/sac2spec_super  \
    src/xc_multi          src/xc_dual         \
    src/stack             src/rotate          \
    src/extractSegments

.PHONY: all debug clean veryclean recurse $(SUBDIRS) help

# ---------- 递归规则 ----------
recurse: $(SUBDIRS)

$(SUBDIRS):
	@echo "========== [ $@ ] =========="
	$(MAKE) -C $@

# ---------- 默认目标：Release ----------
all: ## Build Release (MODE=par|seq)
	@$(info === Building in RELEASE mode, MODE=$(MODE) ===)
	$(MAKE) $(PARFLAG) $(OSYNC) recurse

# ---------- Debug 目标 ----------
debug: export CFLAGS=$(DBG_CFLAGS)
debug: export NVCCFLAGS=$(DBG_NVCCFLAGS)
debug: ## Build Debug (MODE=par|seq)
	@$(info === Building in DEBUG mode, MODE=$(MODE) ===)
	$(MAKE) $(PARFLAG) $(OSYNC) recurse

# ---------- 清理 ----------
clean:
	@for dir in $(SUBDIRS); do \
	  echo ">> Cleaning in $$dir"; \
	  $(MAKE) -C $$dir clean; \
	done

veryclean:
	@for dir in $(SUBDIRS); do \
	  echo ">> Very clean in $$dir"; \
	  $(MAKE) -C $$dir veryclean; \
	done

# ---------- 帮助 ----------
help: ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:  make \033[36m<TARGET>\033[0m [MODE=par|seq]\n\nTargets:\n"} \
	     /^[a-zA-Z0-9_-]+:[^#]*##/{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)

