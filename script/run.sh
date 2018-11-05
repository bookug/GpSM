#!/usr/bin/env bash

# USE the 1-th GPU

#make clean
#make

# NOTICE: make sure that each dataset has exactly 100 queries
data=/home/data/DATASET/

# To run delaunay_n13
result1=delaunay_n13.log/
file1=delaunay_n13.tmp
/bin/rm -rf ${result1}
mkdir ${result1}
for query in `ls /home/data/DATASET/delaunay_n13/query/*`
do
    #echo $query
    file=${query##*/}
    ./GpSM.exe /home/data/DATASET/delaunay_n13/delaunay_n13.g ${query} ${result1}${file%.*}.txt 1 >& ${result1}${file%.*}.log
    grep "match used: " ${result1}${file%.*}.log >> ${file1}
done
awk 'BEGIN{t=0.0}{t = t + $3}END{printf("the average time of answering delaunay_n13 queries: %.2f ms\n", t/100);}' ${file1}
/bin/rm -f $file1
echo "delaunay_n13 ends"

# To run enron
result2=enron.log/
file2=enron.tmp
/bin/rm -rf ${result2}
mkdir ${result2}
for query in `ls /home/data/DATASET/enron/query/*`
do
    file=${query##*/}
    ./GpSM.exe /home/data/DATASET/enron/enron.g ${query} ${result2}${file%.*}.txt 1 >& ${result2}${file%.*}.log
    grep "match used: " ${result2}${file%.*}.log >> ${file2}
done
awk 'BEGIN{t=0.0}{t = t + $3}END{printf("the average time of answering enron queries: %.2f ms\n", t/100);}' ${file2}
/bin/rm -f $file2
echo "enron ends"

# To run gowalla
result3=gowalla.log/
file3=gowalla.tmp
/bin/rm -rf ${result3}
mkdir ${result3}
for query in `ls /home/data/DATASET/gowalla/query/*`
do
    file=${query##*/}
    ./GpSM.exe /home/data/DATASET/gowalla/loc-gowalla_edges.g ${query} ${result3}${file%.*}.txt 1 >& ${result3}${file%.*}.log
    grep "match used: " ${result3}${file%.*}.log >> ${file3}
done
awk 'BEGIN{t=0.0}{t = t + $3}END{printf("the average time of answering enron queries: %.2f ms\n", t/100);}' ${file3}
/bin/rm -f $file3
echo "gowalla ends"

# To run road_central
result4=road_central.log/
file4=road_central.tmp
/bin/rm -rf ${result4}
mkdir ${result4}
for query in `ls /home/data/DATASET/road_central/query/*`
do
    file=${query##*/}
    ./GpSM.exe /home/data/DATASET/road_central/road_central.g ${query} ${result4}${file%.*}.txt 1 >& ${result4}${file%.*}.log
    grep "match used: " ${result4}${file%.*}.log >> ${file4}
done
awk 'BEGIN{t=0.0}{t = t + $3}END{printf("the average time of answering enron queries: %.2f ms\n", t/100);}' ${file4}
/bin/rm -f $file4
echo "road_central ends"

