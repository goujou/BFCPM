# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build

# Path base is the source directory
PKGNAME       = BFMM
PKGDIR        = ../src/$(PKGNAME)
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@(cd source; $(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O))

.PHONY: help Makefile

clean:
	rm -rf $(BUILDDIR)/* || true 
	rm -rf $(SOURCEDIR)/$(PKGNAME).* || true 

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	sphinx-apidoc --separate -f -o $(SOURCEDIR)/ $(PKGDIR)/
	rm $(SOURCEDIR)/modules.rst
	$(SPHINXBUILD) -b html -a "$(SOURCEDIR)" "$(BUILDDIR)"
#	sphinx-apidoc --separate -f -t $(SOURCEDIR)/_templates/ -o $(SOURCEDIR)/ $(PKGDIR)/


