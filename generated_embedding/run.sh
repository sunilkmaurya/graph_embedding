#! /bin/bash

echo "Going to create embeddings with deep walk"
for f in citeseer.edges
do
	echo "File is $f  "
	echo " Loading edgelist :"
	deepwalk --format edgelist --input $f --representation-size 128 --walk-length 40 --window-size 10 --workers 7 --output "$f""_deepwalk.walk"
	#echo "$f""_deepwalk.embeddings"
	echo "Embeddings Created"
done 
echo "Done"
