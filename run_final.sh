#!/bin/bash


python cnn_final.py --dataset mnist --epochs 100 --sample-type uni --lr 0.001 
python cnn_final.py --dataset mnist --epochs 100 --sample-type var --lr 0.001 
python cnn_final.py --dataset mnist --epochs 100 --sample-type obj --lr 0.000464 
python cnn_final.py --dataset mnist --epochs 100 --sample-type grad --lr 0.000216 
python cnn_final.py --dataset cifar --epochs 100 --sample-type uni --lr 0.001 
python cnn_final.py --dataset cifar --epochs 100 --sample-type var --lr 0.001 
python cnn_final.py --dataset cifar --epochs 100 --sample-type obj --lr 0.000464 
python cnn_final.py --dataset cifar --epochs 100 --sample-type grad --lr 0.001 
python fc_final.py --dataset mnist --epochs 100 --sample-type uni --lr 0.001 
python fc_final.py --dataset mnist --epochs 100 --sample-type var --lr 4.641e-5 
python fc_final.py --dataset mnist --epochs 100 --sample-type obj --lr 4.641e-5 
python fc_final.py --dataset mnist --epochs 100 --sample-type grad --lr 2.154e-5 
python fc_final.py --dataset cifar --epochs 100 --sample-type uni --lr 0.0001 
python fc_final.py --dataset cifar --epochs 100 --sample-type var --lr 4.641e-5 
python fc_final.py --dataset cifar --epochs 100 --sample-type obj --lr 4.641e-5 
python fc_final.py --dataset cifar --epochs 100 --sample-type grad --lr 4.641e-5 
