for file in $(ls ../contest_data/)
do
    echo $file
    python Donut.py $file
done