# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

export train_cmd="queue.pl -q cpu_cloudml.q,w1v6.q,cpu.q --mem 2G --max-jobs-run 300"
export decode_cmd="queue.pl -q cpu_cloudml.q,w1v6.q,cpu.q --mem 1G --max-jobs-run 300"
export mkgraph_cmd="queue.pl -q graph.q --mem 11G"
export egs_cmd="queue.pl -q cpu_cloudml.q,w1v6.q,cpu.q --mem 4G --max-jobs-run 300"
export cuda_cmd="queue.pl -q gpu_cloudml_p4_8g.q,v100.q,v100_2.q --mem 5G"

export train_nj=300
export decode_nj=300
