#! /bin/zsh

depths=(2 3 4 5 6)
dataset=eurlex
fanouts=(64 16 8 6 4)
# dataset=wiki10
# fanouts=(176 32 14 8 6)
# dataset=wikiLSHTC
# fanouts=(571 69 24 13 9)
# dataset=WikipediaLarge-500K
# fanouts=(708 80 27 14 9)

run_ids=({1..10})

for fanout in ${fanouts}; do
    for run_id in $run_ids; do
	echo "./depth_exp.sh $dataset $fanout $max_depth $run_id"
    done
done
