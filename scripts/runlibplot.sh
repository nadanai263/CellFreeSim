#!/usr/bin/env fish
vf activate cfs

#echo $PATH # check paths are correct

while read -la line
  python3 libraryplot.py $line
end < modelnames

vf deactivate
