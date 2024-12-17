export CC = gcc
export NVCC = /usr/local/cuda/bin/nvcc
export ARCH = sm_89

SUBDIRS = src/sac2spec src/sac2spec_butter src/sac2spec_cos src/sac2spec_super src/xc_multi src/xc_dual src/stack src/rotate src/extractSegments 

.PHONY: $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	@echo "Making in directory $@"
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do \
		echo "Cleaning in directory $$dir"; \
		$(MAKE) -C $$dir clean; \
	done

veryclean:
	for dir in $(SUBDIRS); do \
		echo "Performing very clean in directory $$dir"; \
		$(MAKE) -C $$dir veryclean; \
	done
