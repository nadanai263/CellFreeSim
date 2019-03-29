#!/usr/bin/env fish
vf activate cfs

#echo $PATH # check paths are correct

while read -la line
  python3 libraryMCMC.py $line
end < modelnames

vf deactivate
