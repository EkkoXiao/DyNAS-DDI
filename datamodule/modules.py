import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from datamodule.dataset import MoleculeDatasetDDIPretrain, MoleculeDatasetDDI
import re

CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"
SPLIT_MARKER = f"|"

def _insert_split_marker(m: re.Match):
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def smiles_handler(text, mol_ph):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)
    text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
    text = escape_custom_split_sequence(text)
    return text, smiles_list


def escape_custom_split_sequence(text):
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)
    
class TrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        
    def __call__(self, batch):
        graphs1, graphs2, texts ,smiles_prompt= zip(*batch)
        graphs1 = self.collater(graphs1)

        smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in smiles_prompt]

        smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)   

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token

        text_tokens = self.tokenizer(text=texts,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        
        graphs2 = self.collater(graphs2)
        
        return graphs1, graphs2, smiles_prompt_tokens, text_tokens   

class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        
    def __call__(self, batch):
        graphs1,graphs2, texts ,smiles_prompt = zip(*batch)

        graphs1 = self.collater(graphs1)
        graphs2 = self.collater(graphs2)

        smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in smiles_prompt]

        smiles_prompt_tokens = self.tokenizer(smiles_prompt, 
                                       return_tensors='pt', 
                                       padding='longest', 
                                       truncation=False, 
                                       return_attention_mask=True)
        

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token
        return graphs1,graphs2, smiles_prompt_tokens, texts       

class DataModuleDDI(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.pretrain_dataset =  MoleculeDatasetDDIPretrain(self.args.train_root, text_max_len, self.prompt, self.args.transform)
        self.train_dataset =  MoleculeDatasetDDI(self.args.train_root, text_max_len, self.prompt, self.args.transform)
        self.val_dataset =  MoleculeDatasetDDI(self.args.valid_root, text_max_len, self.prompt)
        self.test_dataset = MoleculeDatasetDDI(self.args.test_root, text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            )
        else:
            raise NotImplementedError
        return loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        parser.add_argument('--transform', type=str, default=None)
        return parent_parser