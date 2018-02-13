#! /bin/bash

echo "Going to create embeddings with deep struc2vec"
for f in *.edgelist
do
	echo "File is $f  "
	echo " Loading edgelist :"
	python ~/project/embeddings/struc2vec/src/main.py  --input $f --dimensions 128 --walk-length 80 --window-size 5 --workers 5 --num-walk 20 --output "$f""_struc2vec.embeddings" --OPT1 True --OPT2 True --OPT3 True --until-layer 5
	#echo "$f""_struc2vec.embeddings"
	echo "Embeddings Created"
done 
echo "Done"
