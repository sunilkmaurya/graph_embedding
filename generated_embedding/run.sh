#! /bin/bash

echo "Going to create embeddings with deep walk"
for f in blogcatalog.edges
do
	echo "File is $f  "
	echo " Loading edgelist :"
	deepwalk --format edgelist --input $f --representation-size 128 --walk-length 40 --window-size 10 --workers 5 --output "$f""_deepwalk.embeddings"
	#echo "$f""_deepwalk.embeddings"
	echo "Embeddings Created"
done 
echo "Done"
