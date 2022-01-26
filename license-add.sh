#!/bin/bash
set -e

add_license_folder() {
	folder=$1
	echo "adding license to folder $folder"
	for i in $(find $folder -name '*.py');
	do
	  if ! grep -q Copyright $i
	  then
	  	echo "adding license to $i"
	    cat copyright.txt $i > $i.new && mv $i.new $i
	  fi
	done
}

add_license_folder "syne_tune/"
add_license_folder "examples/"
add_license_folder "benchmarking/"
add_license_folder "tst/"

