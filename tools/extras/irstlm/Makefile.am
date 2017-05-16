# not a GNU package. You can remove this line, if
# have all needed files, that a GNU package needs
AUTOMAKE_OPTIONS = foreign

SUBDIRS = src scripts doc 

EXTRA_DIST = README RELEASE Copyright

ACLOCAL_AMFLAGS = -I m4

LN_S=@LN_S@

install-exec-hook:
	cd ${exec_prefix}/ && \
	  ${LN_S} -n -f lib lib64

dist-hook:
	rm -rf `find $(distdir)/doc -type d -name .svn`
