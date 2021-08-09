#!/bin/bash

# It replaces all Git LFS-managed files by pointers.
# Taken from https://stackoverflow.com/a/58316954

lfs_files=($(git lfs ls-files -n))
for file in "${lfs_files[@]}"; do
    git cat-file -e "HEAD:${file}" && git cat-file -p "HEAD:${file}" > "$file"
done

read -p "I'm going to remove .git/lfs/objects. Are you sure? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo Removing
  rm -rf .git/lfs/objects
fi

