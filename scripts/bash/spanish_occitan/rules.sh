#!/bin/bash -v
## This rules weren't found.

test_data=./data/floresp-v2.0-rc.3/floresp-v2.0-rc.3/dev/dev.spa_Latn
output_dir=./results/rule_based/

while read line
  do echo ${line} | apertium apertium spa-oc
done < ${test_data} > ${test_data}/occitan_from_spanish.out
sed -r 's/\*//g' ${output_dir}/occitan_from_spanish.out > ${output_dir}/occitan_from_spanish.posprocess
