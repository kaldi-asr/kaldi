#!/usr/bin/env python
from pylauncher import *
import pylauncher
import sys

import os
import errno

def make_path(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise
		elif not os.path.isdir(path):
			raise

def tail(n, filename):
	import subprocess
	p=subprocess.Popen(['tail','-n',str(n),filename], stdout=subprocess.PIPE)
	soutput,sinput=p.communicate()
	soutput=soutput.split("\n")
	return soutput

def KaldiLauncher(lo, **kwargs):
	import time;
	jobid = JobId()
	debug = kwargs.pop("debug","")
	qdir= os.path.join(lo.qdir, lo.taskname);
	cores = lo.nof_threads;

	ce=SSHExecutor(workdir=qdir, debug=debug, force_workdir=True, catch_output=True)
	ce.outstring="out."
	ce.execstring=lo.taskname + "."

	hostpool=HostPool(hostlist=HostListByName(), commandexecutor=ce )
	
	completion=lambda x:FileCompletion( taskid=x, stamproot="done.", stampdir=qdir)	

	logfiles = list()
	commands = list()
	for q in xrange(lo.jobstart, lo.jobend+1):
		s = "bash " + lo.queue_scriptfile + " " + str(q) 
		commands.append(s)

		logfile = lo.logfile.replace("${PY_LAUNCHER_ID}", str(q))
		logfiles.append(logfile)

	generator=ListCommandlineGenerator(list=commands, cores=cores)
	tasks = TaskGenerator(generator, completion=completion, debug=debug )

	job = LauncherJob( hostpool=hostpool, taskgenerator=tasks, debug=debug,**kwargs)

	job.run()
	#At this point all the .done files should exist and everything should be finalized.
	num_failed=0;
	time.sleep(1); #Lets wait for a while to give the shared fs time to sync
	error_pending=True
	for logfile in logfiles:
		import time
		sched_rate=[0, 0.5, 1, 2, 4, 8, 15, 32 ];
		for delay in sched_rate:
			time.sleep(delay);
			if os.path.isfile(logfile):
				break;
		if not os.path.isfile(logfile):
			sys.stderr.write("ERROR: " + "The following file is missing:\n")
			sys.stderr.write("ERROR: " + "\t" + logfile + "\n")
			sys.stderr.write("ERROR: " + "That means something went wrong, but we don't know what. Try to figure out what and fix it\n");
			sys.exit(-1);
	
		error_pending=True;
		for delay in sched_rate:
			time.sleep(delay);
			
			lines=tail(10, logfile)
			with_status=filter(lambda x:re.search(r'with status (\d+)', x), lines)
		
			if len(with_status) == 0:
				sys.stderr.write("The last line(s) of the log-file " + logfile + " does not seem"
						" to indicate return status as expected\n");
			elif len(with_status) > 1:
				sys.stderr.write("The last line(s) of the log-file " + logfile + " does seem"
						" to indicate multiple return statuses \n");
			else: 
				status_re=re.search(r'with status (\d+)', with_status[0]);
				status=status_re.group(1);
				if status == '0':
					error_pending=False;
				break;
			sys.stderr.write("INFO: Waiting for status in files, sleeping %d seconds\n" %	(delay,))
		if error_pending:
			num_failed+=1;

	if num_failed != 0:
		sys.stderr.write(sys.argv[0] + ": " + str(num_failed) + "/" + str(len(logfiles)) +  " failed \n");
		sys.stderr.write(sys.argv[0] + ": See  " + lo.logfile.replace("${PY_LAUNCHER_ID}", "*" ) + " for details\n");
		sys.exit(-1);

	#Remove service files. Be careful not to remove something that might be needed in problem diagnostics	
	for i in xrange(len(commands)):
		out_file=os.path.join(qdir, ce.outstring+str(i))

		#First, let's wait on files missing (it might be that those are missing
		#just because of slow shared filesystem synchronization
		if not os.path.isfile(out_file):
			import time
			sched_rate=[0.5, 1, 2, 4, 8 ];
			for delay in sched_rate:
				time.sleep(delay);
				if os.path.isfile(out_file):
					break;
			if not os.path.isfile(out_file):
				sys.stderr.write("ERROR: " + "The following file is missing:\n")
				sys.stderr.write("ERROR: " + "\t" + out_file + "\n")
				sys.stderr.write("ERROR: " + "That means something went wrong, but we don't know what. Try to figure out what and fix it\n");
				sys.exit(-1);

		if os.stat(out_file).st_size != 0:
			sys.stderr.write("ERROR: " + "The following file has non-zero size:\n")
			sys.stderr.write("ERROR: " + "\t" + out_file + "\n")
			sys.stderr.write("ERROR: " + "That means something went wrong, but we don't know what. Try to figure out what and fix it\n");
			sys.exit(-1);
		else:
			exec_file=os.path.join(qdir, ce.execstring+str(i))
			done_file=os.path.join(qdir, "done."+str(i))
			if (not os.path.isfile(exec_file) ) or (not os.path.isfile(done_file)):
				sys.stderr.write("ERROR: " + "One of the following files is missing:\n")
				sys.stderr.write("ERROR: " + "\t" + exec_file + "\n")
				sys.stderr.write("ERROR: " + "\t" + done_file + "\n")
				sys.stderr.write("ERROR: " + "\t" + out_file + "\n")
				sys.stderr.write("ERROR: " + "That means something went wrong, but we don't know what. Try to figure out what and fix it\n");
				sys.exit(-1);
			elif os.stat(done_file).st_size != 0:
				sys.stderr.write("ERROR: " + "The following file has non-zero size:\n")
				sys.stderr.write("ERROR: " + "\t" + done_file + "\n")
				sys.stderr.write("ERROR: " + "That means something went wrong, but we don't know what. Try to figure out what and fix it\n");
				sys.exit(-1);
			else:
				os.remove(exec_file)
				os.remove(done_file)
				os.remove(out_file)
	try:
		os.rmdir(qdir) 
	except OSError:
		sys.stderr.write("ERROR: " + "Failed to remove the pylauncher task dir " + qdir + "\n");
		sys.stderr.write("ERROR: " + "Find out what is wrong and fix it\n")
		sys.exit(-1);
	
	#print job.final_report()

class LauncherOpts:
	def __init__(self):
		self.sync=0
		self.nof_threads = 1
		self.qsub_opts = None

		self.jobname=None
		self.jobstart=None
		self.jobend=None
		pass

def CmdLineParser(argv):
	import re;
	sync=0
	qsub_opts=''
	nof_threads=1

	while  len(argv) >= 2 and argv[0].startswith('-'):
		switch = argv.pop(0);

		if switch == '-V':
			qsub_opts += switch + ' ';
		else:
			option = argv.pop(0)
			
			if switch == "-sync" and (option in ['Y', 'y']):
				sync=1;
			qsub_opts += switch + ' ' + option + ' ';
			if switch == "-pe":
				option2 = argv.pop(0);
				qsub_opts += option2 + ' ';
				nof_threads = int(option2);

	#Now we have to parse the JOB specifier 
	jobname = ""
	jobstart = 0
	jobend = 0
	if (re.match( r"^[A-Za-z_]\w*=\d+:\d+$", argv[0])):
		m=re.match( r"^([A-Za-z_]\w*)=(\d+):(\d+)$", argv[0])
		jobname=m.group(1)
		jobstart=int(m.group(2))
		jobend=int(m.group(3))
		argv.pop(0)
	elif(re.match( r"^[A-Za-z_]\w*=\d+$", argv[0])):
		m=re.match( r"^([A-Za-z_]\w*)=(\d+)$", argv[0])
		jobname=m.group(1)
		jobstart=int(m.group(2))
		jobend=int(m.group(2))
		argv.pop(0)
	elif re.match("^.+=.*:.*$", argv[0]):
		print >> sys.stderr, "warning: suspicious JOB argument " + argv[0];

	if jobstart > jobend:
		sys.stderr.write("lonestar.py: JOBSTART("+ str(jobstart) + ") must be lower than JOBEND(" + str(jobend) + ")\n")
		sys.exit(1)

	logfile=argv.pop(0)

	opts=LauncherOpts()
	opts.sync = sync
	opts.nof_threads=nof_threads;
	opts.qsub_opts=qsub_opts
	opts.varname=jobname
	opts.jobstart=jobstart
	opts.jobend=jobend
	opts.logfile=logfile
	
	opts.cmd = escape_cmd(argv);

	return (opts, argv)	

def escape_cmd(argv):
	cmd =""
	for x in argv:
		#print x + " -> ",
		if re.search("^\S+$", x):
			#print " A -> ",
			cmd += x + " "
		elif '"' in x:
			cmd += "'''" + x + "''' "
		else:
			cmd += "\"" + x + "\" "
		#print cmd
	return cmd

def setup_paths_and_vars(opts):
	cwd = os.getcwd()

	if opts.varname and (opts.varname not in opts.logfile ) and (opts.jobstart != opts.jobend):
		print >>sys.stderr, "lonestar.py: you are trying to run a parallel job" \
			"but you are putting the output into just one log file (" + opts.logfile + ")";
		sys.exit(1)

	if not os.path.isabs(opts.logfile):
		opts.logfile = os.path.join(cwd, opts.logfile);
	logfile=opts.logfile

	dir = os.path.dirname(logfile)
	base = os.path.basename(logfile)
	qdir = os.path.join(dir, "q");

	if re.search("log/*q", qdir, flags=re.IGNORECASE):
		qdir = re.sub("log/*q", "/q", qdir, flags=re.IGNORECASE)

	
	queue_logfile= os.path.join(qdir, base)
	if opts.varname:
		queue_logfile = re.sub("\.?"+opts.varname, "", queue_logfile)

	taskname=os.path.basename(queue_logfile)
	taskname = taskname.replace(".log", "");
	if taskname == "":
		print >> sys.stderr, "lonestar.py: you specified the log file name in such form " \
			"that leads to an empty task name ("+logfile + ")";
		sys.exit(1)

	if not os.path.isabs(queue_logfile):
		queue_logfile= os.path.join(cwd, queue_logfile)

	if opts.varname:
		opts.logfile = opts.logfile.replace(opts.varname, "${PY_LAUNCHER_ID}")
		opts.cmd = opts.cmd.replace(opts.varname, "${PY_LAUNCHER_ID}");

	queue_scriptfile=queue_logfile;
	if re.search("\.[a-zA-Z]{1,5}$", queue_scriptfile):
		queue_scriptfile = re.sub("\.[a-zA-Z]{1,5}$", ".sh", queue_scriptfile);
	if not os.path.isabs(queue_scriptfile):
		queue_scriptfile= os.path.join(cwd, queue_scriptfile)

	
	make_path(qdir)
	make_path(dir)

	opts.qdir = qdir
	opts.log_dir = dir
	opts.queue_scriptfile = queue_scriptfile
	opts.queue_logfile = queue_logfile
	opts.taskname = taskname
	
	return opts	


 
def create_scriptfile(scriptname, opts):
	import os
	logfile = opts.logfile
	cmd = opts.cmd
	nof_threads=opts.nof_threads;
	cwd = os.getcwd()
	#print scriptname
	f = open(scriptname, "wb")
	f.write("#!/bin/bash\n")
	f.write("export PY_LAUNCHER_ID=$1; shift;\n")
	f.write("cd " + cwd + "\n")
	f.write(". ./path.sh\n")
	f.write("( echo '#' Running on `hostname`\n")
	f.write("  echo '#' Started at `date`\n")
	f.write("  echo -n '# '; cat <<EOF\n")
	f.write(cmd + "\n")
	f.write("EOF\n")
	f.write(") > " +logfile + "\n")
	f.write("time1=`date +\"%s\"`\n")
	f.write("( " + cmd + ") 2>>" + logfile + " >>" + logfile + " \n")
	f.write("ret=$?\n")
	f.write("time2=`date +\"%s\"`\n")
	f.write("echo '#' Accounting time=$(($time2 - $time1)) threads=" + str(nof_threads) + " >> " + logfile + "\n")

	f.write("echo '#' Finished at `date` with status $ret >>" + logfile + "\n")
	f.write("exit $ret \n")
	f.close()
	

	
if __name__ == "__main__":
	(opts, cmd) = CmdLineParser(sys.argv[1:]);
	setup_paths_and_vars(opts)
	create_scriptfile(opts.queue_scriptfile, opts);

	#pylauncher.ClassicLauncher(["true && sleep 10s", "false || sleep 1s" ], debug="job+host+task+exec+ssh")
	KaldiLauncher(opts, debug="")


