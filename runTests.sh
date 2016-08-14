#!/usr/bin

for t in tests/test*.py; do 
    echo "Running $t..."
    mpiexec -n 8 python $t >& "${t}.log"
    if [ "$?" -ne 0 ]; then
    	echo "*** ERROR encountered when running $t"
    else
    	echo "... Test $t ran successfully"
    fi
done
