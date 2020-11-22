#!/bin/bash

awk '{print 1,$1,$2,$3}' GrC.dat  > positions_tmp.dat
awk '{print 2,$1,$2,$3}' PC.dat   >> positions_tmp.dat
awk '{print 3,$1,$2,$3}' GoC.dat  >> positions_tmp.dat
awk '{print 4,$1,$2,$3}' SC.dat   >> positions_tmp.dat
awk '{print 5,$1,$2,$3}' BC.dat   >> positions_tmp.dat
awk '{print 6,$1,$2,$3}' IO.dat   >> positions_tmp.dat
awk '{print 7,$1,$2,$3}' DCN.dat  >> positions_tmp.dat
awk '{print 8,$1,$2,$3}' Glom.dat >> positions_tmp.dat

cat -n positions_tmp.dat > positions.dat
