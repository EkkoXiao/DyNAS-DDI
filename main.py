import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from datamodule.modules import DataModuleDDI
from model.dynas import DyNAS
from model.nasmodel import GraceModel

os.environ['OPENBLAS_NUM_THREADS'] = '1'
# Medium (bfloat16), High (tensorfloat32), Highest (float32)
torch.set_float32_matmul_precision('medium') 

class MyDDPSpawnStrategy(strategies.DDPSpawnStrategy):
    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)

def main(args):
    pl.seed_everything(args.seed)
    # Load model
    if args.init_checkpoint:
        model = DyNAS.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = DyNAS(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    else:
        model = DyNAS(args)

    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params:', total_params)

    if args.opt_model.find('galactica') >= 0:
        tokenizer = model.blip2opt.opt_tokenizer
    else:
        tokenizer = model.blip2opt.llm_tokenizer

    if args.DDI == True or args.drugbank == True:
        dm = DataModuleDDI(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, tokenizer, args)
    else:
        raise(NotImplementedError)
    

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_last=True, 
                                         save_top_k=-1))
    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'fsdp':
            strategy = strategies.DDPFullyShardedNativeStrategy()
        elif args.strategy_name == 'deepspeed':
            strategy = strategies.DeepSpeedStrategy(stage=3)
        else:
            strategy = MyDDPSpawnStrategy(find_unused_parameters=False)
    else:
        args.devices = eval(args.devices)
        strategy = strategies.SingleDeviceStrategy(device=args.devices)

    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    
    trainer = Trainer.from_argparse_args(args,
                                        callbacks=callbacks,
                                        strategy=strategy,
                                        logger=logger,
                                        )

    if args.mode in {'pretrain', 'ft'}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    elif args.mode == 'test':
        trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, default='ft')
    parser.add_argument('--strategy_name', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser = DyNAS.add_model_specific_args(parser)
    parser = DataModuleDDI.add_model_specific_args(parser)
    parser = GraceModel.add_model_specific_args(parser)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='1')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--double', type=bool, default=False)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--train_root', type=str, default='')
    parser.add_argument('--valid_root', type=str, default='')
    parser.add_argument('--test_root', type=str, default='')
    parser.add_argument('--DDI', action="store_true", default=False)
    parser.add_argument('--drugbank', action="store_true", default=False)
    parser.add_argument('--NAS', action="store_true", default=False)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args

if __name__ == '__main__':
    main(get_args())