#!/bin/bash
# To publish a release 0.1.0, launch this script with
# ./release.sh 0.1.0
#
# This will:
# * build the package
# * update the file version.py that contains the last release
# * commit the release with a tag remotely
# * publish the build package pypi

set -e

rm -rf dist/*

version=$1
version_file=version.py

python setup.py sdist bdist_wheel

echo "$version" > "$version_file"
git commit -am "Release $version"
git tag v$version
git push

# requires a test Pypi and a Pypi account
# checks upload on testpypi first
# twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*

twine upload --verbose dist/*
