#!/bin/sh

# install development dependencies and setup git hook to automatically apply code-formatting

pip install -e .[extra,dev]
# update location of Git hooks from default (.git/hooks) to the versioned folder .devtools/githooks
git config core.hooksPath "githooks"
