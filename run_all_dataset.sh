for FILE in ./configs/*; do
 echo $FILE
 sbatch train_nerf.sh $FILE
done