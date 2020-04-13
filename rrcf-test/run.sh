for i in $( seq 15 28 )
do
python3 test.py $1 $i 1 &
# $1是实验版本， $i 是第几个batch, 1代表是batch大小
done
