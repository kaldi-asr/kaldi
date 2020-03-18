# The `extensions` list should already be in here from `sphinx-quickstart`
extensions = [
    # there may be others here already, e.g. 'sphinx.ext.mathjax'
    'breathe',
    'exhale'
]

# Setup the breathe extension
breathe_projects = {
    "Kaldi": "./xml"
}
breathe_default_project = "Kaldi"

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "rootFileTitle":         "Library API",
    "doxygenStripFromPath":  "..",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleUseDoxyfile": True,
}

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'
