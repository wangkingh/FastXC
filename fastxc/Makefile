SUBDIRS = src/sac2spec src/xc src/rotate src/extractSegments

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
