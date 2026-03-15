import os
from typing import Any, Dict
import torch
from model.blip2opt import Blip2OPT
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import torch.distributed as dist

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    model.load_state_dict(state_dict, strict=True)

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict

class DyNAS(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        return super().on_save_checkpoint(checkpoint)
    
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.reaction_weight = args.reaction_weight
        self.llm_tune = args.llm_tune
        self.previous_state_dict = {}
        self.previous_unchanged_keys = None
        if args.opt_model.find('galactica') >= 0:
            self.blip2opt = Blip2OPT(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, use_nas=args.NAS, args=args)
            self.tokenizer = self.blip2opt.opt_tokenizer
        
        self.save_hyperparameters(args)
    
    def configure_optimizers(self):
        self.trainer.reset_train_dataloader()
        if self.args.NAS:
            encoder_params = []
            ag_params = []
            params = []

            # Group parameters by name
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "graph_encoder" in name and "supernet0" in name:
                        encoder_params.append(param)
                    elif "graph_encoder" in name and "ag" in name:
                        ag_params.append(param)
                    else:
                        params.append(param)

            optimizer = torch.optim.AdamW([
                {"params": params, "lr": self.args.init_lr, "weight_decay": self.args.weight_decay},
                {"params": encoder_params, "lr": self.args.encoder_learning_rate, "weight_decay": self.args.encoder_weight_decay},
                {"params": ag_params, "lr": self.args.arch_learning_rate, "weight_decay": self.args.arch_weight_decay},
            ])

            warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer,
                self.args.max_epochs, 
                self.args.min_lr, 
                self.args.init_lr, 
                warmup_steps, 
                self.args.warmup_lr
            )

            return optimizer
        else:
            warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                self.args.init_lr, 
                weight_decay=self.args.weight_decay
            )
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer, 
                self.args.max_epochs, 
                self.args.min_lr, 
                self.args.init_lr, 
                warmup_steps, 
                self.args.warmup_lr
            )
            return optimizer

    def test_epoch_end(self, outputs):
        list_predictions, list_targets = zip(*outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]

        if self.trainer.world_size > 1:  # Only gather when in distributed mode
            all_predictions = [None for _ in range(self.trainer.world_size)]
            all_targets = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)

            if self.global_rank == 0:
                all_predictions = [i for ii in all_predictions for i in ii]
                all_targets = [i for ii in all_targets for i in ii]
                self.save_predictions_test(all_predictions, all_targets)
        else:
            # For single GPU, no need for dist.all_gather_object
            self.save_predictions_test(predictions, targets)

    def save_predictions_test(self, predictions, targets):
        ## print("##### Save Prediction Test #####")
        if self.args.DDI or self.args.drugbank:
            assert len(predictions) == len(targets)
            predict_dir = 'result/' + self.args.filename
            os.makedirs(predict_dir, exist_ok=True)
            predict_file = os.path.join(predict_dir, f'predictions_epoch={self.current_epoch + 1}.txt')
            with open(predict_file, 'w', encoding='utf8') as f:
                for (decoded, prob, output), t in zip(predictions, targets):
                    line = {'prediction': decoded, 'probability': prob, 'text': output, 'target': t}
                    f.write(json.dumps(line, ensure_ascii=True) + '\n')
        else:
            assert len(predictions) == len(targets)
            predict_dir = 'result/' + self.args.filename
            os.makedirs(predict_dir, exist_ok=True)
            predict_file = os.path.join(predict_dir, f'predictions_epoch={self.current_epoch + 1}.txt')
            with open(predict_file, 'w', encoding='utf8') as f:
                for p, t in zip(predictions, targets):
                    line = {'prediction': p, 'target': t}
                    f.write(json.dumps(line, ensure_ascii=True) + '\n')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if self.args.DDI == True:
            graphs1, graphs2, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                output_scores=True
            )
            max_probs, max_indices = torch.nn.functional.softmax(predictions["scores"][0], dim=-1).max(dim=-1)
            max_probs, max_indices = max_probs[::self.num_beams], max_indices[::self.num_beams]
            decoded_tokens = self.tokenizer.batch_decode(max_indices.unsqueeze(-1))

            for i, (token, prob) in enumerate(zip(decoded_tokens, max_probs.tolist())):
                print(f"Sample {i}: Token: {token}, Probability: {prob:.4f}")

            predictions = [(token, prob) for token, prob in zip(decoded_tokens, max_probs.tolist())]
            outputs = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            predictions = [(token, prob, output) for (token, prob), output in zip(predictions, outputs)]
            return predictions, texts
        else:
            graphs1, graphs2, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
            return predictions, texts
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if self.args.DDI == True:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            graphs1, graphs2, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                output_scores=True
            )
            max_probs, max_indices = torch.nn.functional.softmax(predictions["scores"][0], dim=-1).max(dim=-1)
            max_probs, max_indices = max_probs[::self.num_beams], max_indices[::self.num_beams]
            decoded_tokens = self.tokenizer.batch_decode(max_indices.unsqueeze(-1))

            predictions = [(token, prob) for token, prob in zip(decoded_tokens, max_probs.tolist())]
            outputs = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            predictions = [(token, prob, output) for (token, prob), output in zip(predictions, outputs)]
            return predictions, texts
        elif self.args.drugbank == True:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            graphs1, graphs2, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                output_scores=True
            )

            cat_token_ids = [self.tokenizer.convert_tokens_to_ids(f"<CAT{i:02d}>") for i in range(86)]
            
            cat_logits = predictions["scores"][0][:, cat_token_ids]
            cat_probs = torch.nn.functional.softmax(cat_logits, dim=-1) 

            prob_sums = cat_probs.sum(dim=-1) 
                 
            max_probs, max_cat_indices = cat_probs.max(dim=-1)
            max_probs, max_cat_indices = max_probs[::self.num_beams], max_cat_indices[::self.num_beams]
            
            decoded_tokens = [f"<CAT{idx:02d}>" for idx in max_cat_indices.tolist()]
            
            all_probs = cat_probs[::self.num_beams].tolist()  # shape: [batch_size, 86]
            
            predictions = [(token, probs) for token, prob, probs in zip(decoded_tokens, max_probs.tolist(), all_probs)]
            
            outputs = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            predictions = [(token, probs, output) for (token, probs), output in zip(predictions, outputs)]
            return predictions, texts
        else:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            graphs1, graphs2, prompt_tokens, texts = batch
            samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
            return predictions, texts

    def validation_epoch_end(self, outputs):
        if self.current_epoch != 0:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            caption_outputs = outputs[1]
            list_predictions, list_targets = zip(*caption_outputs)
            predictions = [i for ii in list_predictions for i in ii]
            targets = [i for ii in list_targets for i in ii]
            if self.trainer.world_size > 1:  # Only gather when in distributed mode
                all_predictions = [None for _ in range(self.trainer.world_size)]
                all_targets = [None for _ in range(self.trainer.world_size)]
                dist.all_gather_object(all_predictions, predictions)
                dist.all_gather_object(all_targets, targets)

                if self.global_rank == 0:
                    all_predictions = [i for ii in all_predictions for i in ii]
                    all_targets = [i for ii in all_targets for i in ii]
                    self.save_predictions_test(all_predictions, all_targets)
            else:
                self.save_predictions_test(predictions, targets)

    def training_step(self, batch, batch_idx):
        batch_size = batch[-1].input_ids.size(0)
        loss = self.blip2opt(batch) 

        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        self.log("loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)

        return loss['loss']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Blip2Stage2")
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', action='store_true', default=True)
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # OPT Open Pre-trained Transformer
        parser.add_argument('--opt_model', type=str, default="facebook/galactica-1.3b")
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_len', type=int, default=256)
        parser.add_argument('--min_len', type=int, default=8)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--peft_dir', type=str, default='')
        parser.add_argument('--save_every_n_epochs', type=int, default=0)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)
        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)
        # optimization
        parser.add_argument('--reaction_weight', type=float, default=1.0)
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=10)
        return parent_parser


