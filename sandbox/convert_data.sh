#! /bin/zsh

datasets=(eurlex)


for dataset in ${datasets}; do
    echo ${dataset}
    dir="data/${dataset}"

    echo ${dir}

    if [ ! -f ${dir}/trn_X_Y.txt ]; then
	perl ../tools/convert_format.pl\
	     ${dir}/${dataset}_train.txt\
	     ${dir}/trn_X_Xf.txt\
	     ${dir}/trn_X_Y.txt
    fi

    if [ ! -f ${dir}/tst_X_Y.txt ]; then    
	perl ../tools/convert_format.pl\
	     ${dir}/${dataset}_test.txt\
	     ${dir}/tst_X_Xf.txt\
	     ${dir}/tst_X_Y.txt
    fi
done
