.PHONY: doc tex-doc

tex-doc: doc
	$(MAKE) -C doc/latex

doc:
	doxygen Doxyfile

help:
	@echo "This Makefile builds documentation. Valid targets are:"
	@echo "... doc "
	@echo "... tex-doc"
	@echo ""
	@echo "See README in order to build the code."

