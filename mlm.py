from optimus.encoder_layer import EncoderLayer
from optimus.optimus_embedding import OptimusEmbedding
import torch
from torch.optim import AdamW
import torch.nn as nn
import pytorch_lightning as pl
from datasets import load_from_disk
from torch.utils.data.dataloader import DataLoader

from optimus.heads.mlm_head import MlmHead

MODEL_SIZE = 100
SEQ_LEN = 128

class OptimusForMlm(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        
        self.embedding = OptimusEmbedding(10_000,  MODEL_SIZE, SEQ_LEN)
        self.encoder = EncoderLayer(4, MODEL_SIZE, 48, SEQ_LEN, 2*MODEL_SIZE, dropout=0.2)
        self.head = MlmHead(model_size=MODEL_SIZE, vocab_size=10_000)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        embedded = self.embedding(x)
        encoded, attn_map = self.encoder(embedded)
        logits = self.head(encoded)

        return logits, attn_map

    def training_step(self, batch, batch_idx):
        
        out, _ = self.forward(batch['ids'])

        loss = self.criterion(out, batch['labels'])

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters())

    
dset = load_from_disk('wiki128')
# tokenizer = 
loader = DataLoader(dset, batch_size=8)

model = OptimusForMlm()
trainer = pl.Trainer(gpus=None)

trainer.fit(model, train_dataloader=loader)

    