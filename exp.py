from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer
from optimus.optimus_embedding import OptimusEmbedding
from optimus.encoder_layer import EncoderLayer
from optimus.heads.classification_head import ClassificationHead
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim
from pytorch_lightning.metrics import Fbeta,Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

SEQ_LEN=256
BATCH_SIZE=16
VOCAB_SIZE=tokenizer.vocab_size
MODEL_SIZE = 64
SEQ_LEN = 128


# dataset_train = load_dataset('imdb', split='train')
# dataset_test = load_dataset('imdb', split='test')

# dataset_train = dataset_train.map(
#     lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=SEQ_LEN), batched=True, batch_size=128
# )
# dataset_train.save_to_disk('tokenized_train')
# dataset_test = dataset_test.map(
#     lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=SEQ_LEN), batched=True, batch_size=128
# )
# dataset_test.save_to_disk('tokenized_test')

dataset_train = load_from_disk('tokenized_train')
dataset_test = load_from_disk('tokenized_test')

dataset_train.set_format('pt', columns=['input_ids', 'label'])
dataset_test.set_format('pt', columns=['input_ids', 'label'])

dl_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dl_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)


class Transformer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.embedding = OptimusEmbedding(30522,  MODEL_SIZE, SEQ_LEN)
        self.encoder = EncoderLayer(4, MODEL_SIZE, 48, SEQ_LEN, 2*MODEL_SIZE, dropout=0.2)
        self.head = ClassificationHead(MODEL_SIZE, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.f1 = Fbeta(num_classes=2)

    def forward(self, x):
        embedded = self.embedding(x)
        encoded, attn_map = self.encoder(embedded)
        logits = self.head(encoded)

        return logits, attn_map

    def training_step(self, batch, batch_idx):
        x = batch['input_ids']
        y = batch['label']

        out, attn_map = self.forward(x)

        loss = self.criterion(out, y)

        pred = torch.argmax(out, -1)

        self.log('train_acc', self.accuracy(pred, y), on_step=False, on_epoch=True)
        self.log('train_f1', self.f1(pred, y), on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        x = batch['input_ids']
        y = batch['label']

        out, attn_map = self.forward(x)

        loss = self.criterion(out, y)

        pred = torch.argmax(out, -1)
        
        self.log('val_acc', self.accuracy(pred, y), on_step=False, on_epoch=True)
        self.log('val_f1', self.f1(pred, y), on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

model = Transformer()

checkpoint_callback =  ModelCheckpoint(
    monitor='val_acc', dirpath='checkpoints', save_top_k=3, filename='tr-{epoch:02d}-{val_acc:.2f}',
    mode='auto'
)

trainer = pl.Trainer(
    gpus=[0],
    gradient_clip_val=1.0,
    logger=TensorBoardLogger(
        'tb'
    )
    # val_check_interval=0.


)

trainer.fit(model, train_dataloader=dl_train, val_dataloaders=dl_test)