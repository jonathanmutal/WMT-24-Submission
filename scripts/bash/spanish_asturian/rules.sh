#!/bin/bash -v

test_data=./data/floresp-v2.0-rc.3/floresp-v2.0-rc.3/dev/dev.spa_Latn
output_dir=./results/rule_based/

while read line
  do echo ${line} | apertium spa-ast
done < ${test_data} > ${test_data}/asturian_from_spanish.out
sed -r 's/\*//g' ${output_dir}/asturian_from_spanish.out > ${output_dir}/asturian_from_spanish.posprocess
