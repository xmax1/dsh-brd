
from pathlib import Path

from pydantic import BaseModel


class UserConfig(BaseModel):
  name: str
  secret: str

class Pyfig(BaseModel):
  user:                 str = UserConfig
  project:          str = None
  env:              str = None
  run_name:         Path = None
  exp_name: str	= None
  exp_id: str	= None
  group_exp: bool	= False
  lo_ve_path: str = None
  mode: str	= None
  multimode: str = None
  debug: bool = False
  seed: int = 0
  _dtype_str: str = 'float32'

	run_id:	str = None
	submit: bool = False
	dtype: str = None
	device: str 	= None

	n_log_metric:	int = 100
	n_log_state: int = 4
	is_logging_process: bool = False

	cudnn_benchmark: 	bool 	= True


	zweep: str = ''

	n_default_step: 	int 	= 10
	n_train_step:   	int   	= 0
	n_pre_step:    		int   	= 0
	n_eval_step:        int   	= 0
	n_opt_hypam_step:   int   	= 0
	n_max_mem_step:     int   	= 0


	data_tag: str		= 'data'
	
	max_mem_alloc_tag: str = 'max_mem_alloc'
	opt_obj_all_tag: str = 'opt_obj_all'
	opt_obj_tag: str = 'opt_obj'
		
	pre_tag: str = 'pre'
	train_tag: str = 'train'
	eval_tag: str = 'eval'
	opt_hypam_tag: str = 'opt_hypam'
		
	v_cpu_d_tag: str = 'v_cpu_d'
	c_update_tag: str = 'c_update'
		
	lo_ve_path_tag: str = 'lo_ve_path'
	gather_tag: str = 'gather'
	mean_tag: str = 'mean'

	ignore_f: list = ['commit', 'pull', 'backward']
	ignore_p: list = ['parameters', 'scf', 'tag', 'mode_c']
	ignore: list = ['ignore', 'ignore_f', 'ignore_c'] + ignore_f + ignore_p
	ignore += ['d', 'cmd', 'sub_ins', 'd_flat', 'repo', 'name', 'base_d', 'c_init', 'p']
	base_d: dict = None
	c_init: dict = None
	run_debug_c: bool = False
	run_sweep: bool = False


	@property
	def n_step(ii):
		n_step = dict(
			train		= ii.n_train_step, 
			pre			= ii.n_pre_step, 
			eval		= ii.n_eval_step, 
			opt_hypam	= ii.n_opt_hypam_step, 
			max_mem		= ii.n_max_mem_step
		).get(ii.mode)
		if not n_step: 
			n_step = ii.n_default_step
		return n_step

	@property
	def dtype(ii): 
		import torch
		return dict(float64= torch.float64, float32= torch.float32, cpu= 'cpu')[ii._dtype_str]

	@dtype.setter
	def dtype(ii, val):
		if val is not None:
			ii._dtype_str = str(val).split('.')[-1]

	cudnn_benchmark: 	bool 	= True

	n_log_metric:		int  	= 50
	n_log_state:		int  	= 1

	@property
	def is_logging_process(ii: PyfigBase):
		return ii.mode==ii.opt_hypam_tag or ii.dist.head or ii.dist.rank==-1

	opt_obj_key:		str		= 'e'
	opt_obj_op: 	Callable 	= property(lambda _: lambda x: x.std())

	class data(DataBase):
		n_b: int = 128
		loader_n_b: int = 1
		
class model(ModelBase):
  compile_ts: 	bool	= False
  compile_func:	bool	= False
  optimise_ts:	bool	= False
  optimise_aot:	bool 	= False
  with_sign:      bool    = False
  functional: 	bool	= True

  terms_s_emb:    list    = ['ra', 'ra_len']
  terms_p_emb:    list    = ['rr', 'rr_len']
  ke_method:      str     = 'grad_grad'
  n_sv:           int     = 32
  n_pv:           int     = 32
  n_fb:           int     = 3
  n_det:          int     = 4
  n_final_out:	int     = 1
  
  n_fbv:          int     = property(lambda _: _.n_sv*3+_.n_pv*2)

class opt(PyfigBase.opt):
  available_opt:  list 	= ['AdaHessian', 'RAdam', 'Apollo', 'AdaBelief', 'LBFGS', 'Adam', 'AdamW']
  
  opt_name: 		str		= 'AdamW'
  
  lr:  			float 	= 1e-3
  init_lr: 		float 	= 1e-3

  betas:			tuple	= (0.9, 0.999)
  beta: 			float 	= 0.9
  warm_up: 		int 	= 100
  eps: 			float 	= 1e-4
  weight_decay: 	float 	= 0.0
  hessian_power: 	float 	= 1.0
  
class scheduler(PyfigBase.scheduler):
  sch_name: 	str		= 'ExponentialLR'
  sch_max_lr:	float 	= 0.01
  sch_epochs: int 	= 1
  sch_gamma: 	float 	= 0.9999
  sch_verbose: bool 	= False

class sweep(Optuna):
  sweep_name: 	str		= 'study'	
  n_trials: 		int		= 20
  parameters: 	dict 	= dict(
    opt_name		=	Param(values=['AdaHessian', 'RAdam'], dtype=str),
    hessian_power	= 	Param(values=[0.5, 0.75, 1.], dtype=float, condition=['AdaHessian',]),
    weight_decay	= 	Param(domain=(0.0001, 1.), dtype=float, condition=['AdaHessian',]),
    lr				=	Param(domain=(0.0001, 1.), log=True, dtype=float),
  )


### DIST ### 
class DistBase(PlugIn):
	
	dist_name: 		str = 'Base'
	n_launch: 		int = 1
	n_worker:   	int = property(lambda _: _.p.resource.n_gpu)
	ready: 			bool = True
	sync_every: 	int = 1

	rank_env_name: 	str		= 'RANK'
	rank: 			int 	= property(lambda _: int(os.environ.get(_.rank_env_name, '-1')))
	head: 			bool 	= property(lambda _: _.rank==0)
	gpu_id: 		str		= property(lambda _: ''.join(run_cmds(_._gpu_id_cmd, silent=True)).split('.')[0])
	dist_id: 		str 	= property(lambda _: _.gpu_id + '-' + hostname.split('.')[0])
	pid: 			int = property(lambda _: _.rank)

	_gpu_id_cmd:	str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'

class SingleProcess(DistBase):
	
	dist_name: str = 'SingleProcess'
	sync_every: int = property(lambda _: 0) # no sync
	head: int = property(lambda _: True)
	launch_cmd: Callable 	= property(lambda _: 
		lambda node_i, submit_i, cmd: 
		f'\nexport RANK={submit_i} \
		\necho $SLURM_JOB_NODELIST \
		\nsrun --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --ntasks=1 --exclusive --label python -u {_.p.run_name} {cmd} '
	)

class Naive(DistBase):
	
	n_launch:       int = property(lambda _: _.p.resource.n_gpu)
	n_worker:       int = property(lambda _: _.p.resource.n_gpu)
	dist_name: 	str		= 'Naive'
	sync_every:  int 	= 5
	head: 		bool 	= property(lambda _: _.rank==0 or (_.p.opt_hypam_tag in _.p.mode))
	nsync:          int = 0

	launch_cmd: Callable 	= property(lambda _: 
		lambda node_i, submit_i, cmd: 
		f'\nexport RANK={submit_i} \
		\necho $SLURM_JOB_NODELIST \
		\nsrun --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --ntasks=1 --exclusive --label python -u {_.p.run_name} {cmd} '
	)

class HFAccelerate(DistBase):
  sync_every:  int 	= 5
  dist_name: str = 'HFAccelerate'

  class dist_c(PlugIn):
    multi_gpu = True
    machine_rank = property(lambda _: '0')
    # same_network = True
    main_process_port = property(lambda _: find_free_port()) # 
    num_processes =  property(lambda _: str(_.p.p.resource.n_gpu))
    num_machines =  property(lambda _: str(_.p.p.resource.n_node))
    num_cpu_threads_per_process = property(lambda _: str(_.p.p.resource.n_thread_per_process))

  launch_cmd:	Callable  	= property(lambda _: 
    lambda node_i, submit_i, cmd: 
    f'accelerate launch {dict_to_cmd(_.dist_c.d, exclude_false=True)} {_.p.run_name} \ "{cmd}" '
  ) # ! backslash must come between run.py and cmd and "cmd" needed

  # internal
  ready: bool = property(lambda _: _.accel is not None)
  accel: accelerate.Accelerator = None
  split_batches: bool		= False  # if True, reshapes data to (n_gpu, batch_size//n_gpu, ...)
  mixed_precision: str	= 'no' # 'fp16, 'bf16'

  class hostfile(PlugIn):
    slots = 10 # !!! max per resource fix
    ip = property(lambda _: dict(
      ip0 = _.slots,
      ip1 = _.slots,
    ))
### END DIST ###



    
### LOGGER ###
class LoggerBase(PlugIn):
	
	run = None
	
	job_type:		str		= ''		
	log_mode: 		str		= ''
	log_run_path:	str 	= ''
	log_run_type: 	str		= property(lambda _: f'groups/{_.p.exp_name}/workspace') 
	log_run_url: 		str		= property(lambda _: f'{_.p.project}/{_.log_run_type}')

class Wandb(PlugIn):
  
  run = None
  entity:			str		= property(lambda _: _.p.project)
  program: 		Path	= property(lambda _: Path( _.p.paths.project_dir, _.p.run_name))
  
  job_type:		str		= ''		
  log_mode: 		str		= ''
  log_run_path:	str 	= ''
  log_run_type: 	str		= property(lambda _: f'groups/{_.p.exp_name}/workspace') 
  log_run_url: 		str		= property(lambda _: f'https://wandb.ai/{_.entity}/{_.p.project}/{_.log_run_type}')
### END LOGGER ###

    
    
### RESOURCE ###

ResourceBase(PlugIn):
	n_gpu: int = 0
	n_node: int = 1
	n_thread_per_process: int = 1

	def cluster_submit(ii, job: dict):
		return job
	
	def device_log_path(ii, rank=0):
		if not rank:
			return ii.p.paths.exp_dir/(str(rank)+"_device.log") 
		else:
			return ii.p.paths.cluster_dir/(str(rank)+"_device.log")

class Niflheim(ResourceBase):

	n_gpu: int 		= 1
	n_node: int 	= property(lambda _: int(math.ceil(_.n_gpu/10))) 
	n_thread_per_process: int = property(lambda _: _.slurm_c.cpus_per_gpu)

	architecture:   	str 	= 'cuda'
	nifl_gpu_per_node: 	int  = property(lambda _: 10)

	job_id: 		str  	= property(lambda _: os.environ.get('SLURM_JOBID', 'No SLURM_JOBID available.'))  # slurm only

	_pci_id_cmd:	str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'
	pci_id:			str		= property(lambda _: ''.join(run_cmds(_._pci_id_cmd, silent=True)))

	n_device_env:	str		= 'CUDA_VISIBLE_DEVICES'
	# n_device:       int     = property(lambda _: sum(c.isdigit() for c in os.environ.get(_.n_device_env, '')))
	n_device:       int     = property(lambda _: len(os.environ.get(_.n_device_env, '').replace(',', '')))


	class slurm_c(PlugIn):
		export			= 'ALL'
		cpus_per_gpu    = 8				# 1 task 1 gpu 8 cpus per task 
		partition       = 'sm3090'
		time            = '0-00:10:00'  # D-HH:MM:SS
		nodes           = property(lambda _: str(_.p.n_node)) 			# (MIN-MAX) 
		gres            = property(lambda _: 'gpu:RTX3090:' + str(min(10, _.p.n_gpu)))
		ntasks          = property(lambda _: _.p.n_gpu if int(_.nodes)==1 else int(_.nodes)*80)
		job_name        = property(lambda _: _.p.p.exp_name)
		output          = property(lambda _: _.p.p.paths.cluster_dir/'o-%j.out')
		error           = property(lambda _: _.p.p.paths.cluster_dir/'e-%j.err')

	# mem_per_cpu     = 1024
	# mem				= 'MaxMemPerNode'
	# n_running_cmd:	str		= 'squeue -u amawi -t pending,running -h -r'
	# n_running:		int		= property(lambda _: len(run_cmds(_.n_running_cmd, silent=True).split('\n')))	
	# running_max: 	int     = 20

### END RESOURCE ###
