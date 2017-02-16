# Example MPI LoadLeveler Job file
# @ shell = /bin/bash
#
# @ job_name = testRma
#
# @ job_type = parallel
#
# @ wall_clock_limit     = 0:20:00
#
# @ account_no = hpcf
#
# @ output               = $(job_name).$(schedd_host).$(jobid).o
# @ error                = $(job_name).$(schedd_host).$(jobid).e
# @ notification         = never
# @ class                = General
# @ node = 1
# @ tasks_per_node = 16 
# @ network.MPI = sn_all,not_shared,US
# @ task_affinity = core(1)
#
# @ queue
. /etc/profile
module load python/2.7.5
export PYTHONPATH=/opt/niwa/mpi4py/AIX/2.0.0/lib/python2.7/site-packages/:$PYTHONPATH
echo "Starting at `date`"
poe python exLaplacian3d.py 256 10
echo "Ending at `date`"

