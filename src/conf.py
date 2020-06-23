# -*- coding: utf-8 -*-
import sphinx_rtd_theme
import textwrap

project = 'Kaldi'
copyright = '2009-2020, Kaldi Authors'
author = 'Kaldi Authors'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'breathe',
    'exhale',
]

source_suffix = '.rst'

html_theme = 'sphinx_rtd_theme'

pygments_style = 'sphinx'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
smartquotes = False
html_show_sourcelink = True

html_theme_options = {
    'collapse_navigation': False,
}

# Setup the breathe extension
breathe_projects = {'Kaldi': './xml'}
breathe_default_project = 'Kaldi'

doc_dirs = [
    'base'
#    'chain',
#    'cudadecoder',
#    'cudafeat',
#    'cudamatrix',
#    'decoder',
#    'fstext',
#    'gmm',
#    'hmm',
#    'itf',
#    'ivector',
#    'kws',
#    'lat',
#    'lm',
#    'matrix'
#    'nnet',
#    'nnet2',
#    'nnet3',
#    'online',
#    'online2',
#    'probe',
#    'rnnlm',
#    'sgmm2',
#    'tfrnnlm',
#    'transform',
#    'tree',
#    'util',
]

doc_exclude_patterns = []
for d in doc_dirs:
    pattern = '{0}/*.cc {0}/*-inl.h {0}/*test.cc'.format(d)
    doc_exclude_patterns.append(pattern)

doc_dirs.append('doc')
# yapf: disable
# Setup the exhale extension
exhale_args = {
    # These arguments are required
    'containmentFolder': './api',
    'rootFileName': 'api.rst',
    'rootFileTitle': 'Kaldi C++ API',
    'doxygenStripFromPath': '../src',
    # Suggested optional arguments
    'createTreeView': True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # 'treeViewIsBootstrap': True,
    'exhaleExecutesDoxygen': True,
    'exhaleDoxygenStdin': textwrap.dedent('''
        INPUT           = {doc_dirs}
        EXCLUDE         = {doc_exclude_patterns}
        EXAMPLE_PATH    = doc
        EXTRACT_ALL     = yes
    '''.format(doc_dirs=' '.join(doc_dirs),
               doc_exclude_patterns=' '.join(doc_exclude_patterns)))
}
# yapf: enable

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'

