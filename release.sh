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
set -x

rm -rf dist/*

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters, you should pass the version when calling this script, for instance `bash release.sh 0.12`"
fi

version=$1
version_file=syne_tune/version.py
echo \"$version\" > "$version_file"

python setup.py sdist bdist_wheel

git checkout -b $version
git commit -am "Release $version"
git tag v$version
git push --set-upstream origin $version

# requires a test Pypi and a Pypi account
# checks upload on testpypi first
# twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*

twine upload --verbose dist/*
