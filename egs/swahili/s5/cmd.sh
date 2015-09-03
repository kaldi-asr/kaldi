# JHU cluster options
export train_cmd="queue.pl -l arch=*64*"
export decode_cmd="queue.pl -l arch=*64* -l ram_free=4G,mem_free=4G"
export cuda_cmd="..."
export mkgraph_cmd="queue.pl -l arch=*64* ram_free=4G,mem_free=4G"
