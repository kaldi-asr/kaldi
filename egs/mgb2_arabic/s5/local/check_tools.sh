#!/bin/bash

# check whether bs4 and lxml is installed
if ! python3 -c "import bs4" 2>/dev/null; then
  echo "$0: BeautifulSoup4 not installed, you can install it by 'pip install beautifulsoup4' if you prefer to use python to process xml file" 
  exit 1;
fi

if ! python3 -c "import lxml" 2>/dev/null; then
  echo "$0: lxml not installed, you can install it by 'pip install lxml' if you prefer to use python to process xml file"
  exit 1;
fi

echo "both BeatufileSoup4 and lxml are installed in python"
exit 0
