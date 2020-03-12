for i in $( seq 15 28 )
do
python3 test.py $1 $i 1 &
done
