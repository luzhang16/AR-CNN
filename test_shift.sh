#!bin/bash
set -x

session=0
epoch=3
checkpoint=17783
base=0

# S0
for i in $(seq -10 10)
do
(i=$i);
#echo "x:"$i "y:0"
python test_net.py --sx $i --sy 0 --dataset kaist --net vgg16 --checksession $session --checkepoch $epoch --checkpoint $checkpoint --reasonable --cuda
done

# S45
for i in $(seq -10 10)
do
(i=$i);
echo "x:"$i "y:$i"
python test_net.py --sx $i --sy $i --dataset kaist --net vgg16 --checksession $session --checkepoch $epoch --checkpoint $checkpoint --reasonable --cuda
done

# S90
for i in $(seq -10 10)
do
(i=$i);
echo "x:"0 "y:$i"
python test_net.py --sx 0 --sy $i --dataset kaist --net vgg16 --checksession $session --checkepoch $epoch --checkpoint $checkpoint --reasonable --cuda
done

# S135
for i in $(seq -10 10)
do
(j=$base-$i);
echo "x:"$i "y:$j"
python test_net.py --sx $i --sy $j --dataset kaist --net vgg16 --checksession $session --checkepoch $epoch --checkpoint $checkpoint --reasonable --cuda
done


# surface plot
: << !
for i in $(seq -6 6) 
do
(i=$i);
for j in $(seq -6 6)
do
(j=$j);
python test_net_kaist.py --sx $i --sy $j --dataset kaist --net vgg16 --checksession $session --checkepoch $epoch --checkpoint $checkpoint --reasonable --cuda
done
done
!


