# Introduction

We use [Sphinx](https://www.sphinx-doc.org/en/master/) to generate
documentation for kaldi pybind. This document describes the steps
for **developers** to build the environment for generating
documentation.

End users are advised to
visit [TODO(fangjun):fill this](https://kaldi-asr.org/doc/)
for pre-generated documentation.


# Install Sphinx

First, install `sphinx` via `pip`

```sh
pip install -U sphinx
```

To check that `sphinx` is installed successfully, the
command `sphinx-build --version` should print the version
of the installed `sphinx`.

We will use the [theme](https://sphinx-rtd-theme.readthedocs.io/en/stable/)
from <http://www.readthedocs.org/>, which can be installed by

```sh
pip install sphinx_rtd_theme
```

# Setup the template

Run `sphinx-quickstart` to generate the following files and directories:

```
.
├── _build
├── conf.py
├── index.rst
├── make.bat
├── Makefile
├── _static
└── _templates

3 directories, 4 files
```

# Build the documentation

```bash
make html
```

The html files are generated in the directory `_build/html`.
Copy all the files in that directory to the server which is
hosting html pages.

To view the generated documentation in a web browser locally:

```bash
cd _build/html
python3 -m http.server
```

It should print `Serving HTTP on 0.0.0.0 port 8000`.
Then go to your browser and enter `http://localhost:8000`
or `http://<your-server-ip>:8000` to view the
generated documentation.
