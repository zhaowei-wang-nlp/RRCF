start=`expr $1 \* $2`
end=`expr $start + $2`
c=0
for file in $(ls ../contest_data/)
do
    if [ $c -ge $start -a $c -lt $end ]
    then
        echo $file
	python3 Donut.py $file $1
    fi
c=`expr $c + 1`
done
