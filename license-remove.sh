#!/bin/bash
# Script that removes the exact multi-line copyright header from all Python files

set -e

remove_copyright_folder() {
  folder=$1
  echo "Removing copyright from folder $folder"
  for i in $(find $folder -name '*.py');
	do
	  echo $i
	  if grep -q Copyright $i
	  then
	  	echo "removing license from $i"
	    sed -i '' 1,12d "$i"
	  fi
	done
}


remove_copyright_folder "syne_tune/"
remove_copyright_folder "examples/"
remove_copyright_folder "benchmarking/"
remove_copyright_folder "tst/"

