#!/bin/bash


python fc_test.py --dataset cifar --sample-type grad --cross-val --lr 4.641e-5
python cnn_test.py --dataset mnist --sample-type uni --cross-val --lr 0.001
python cnn_test.py --dataset mnist --sample-type var --cross-val --lr 0.001
python cnn_test.py --dataset mnist --sample-type obj --cross-val --lr 0.000464
python cnn_test.py --dataset mnist --sample-type grad --cross-val --lr 0.000216
python cnn_test.py --dataset cifar --sample-type uni --cross-val --lr 0.001
python cnn_test.py --dataset cifar --sample-type var --cross-val --lr 0.001
python cnn_test.py --dataset cifar --sample-type obj --cross-val --lr 0.000464
python cnn_test.py --dataset cifar --sample-type grad --cross-val --lr 0.001
python fc_test.py --dataset mnist --sample-type uni --cross-val --lr 0.001
python fc_test.py --dataset mnist --sample-type var --cross-val --lr 4.641e-5
python fc_test.py --dataset mnist --sample-type obj --cross-val --lr 4.641e-5
python fc_test.py --dataset mnist --sample-type grad --cross-val --lr 2.154e-5
python fc_test.py --dataset cifar --sample-type uni --cross-val --lr 0.0001
python fc_test.py --dataset cifar --sample-type var --cross-val --lr 4.641e-5
python fc_test.py --dataset cifar --sample-type obj --cross-val --lr 4.641e-5
