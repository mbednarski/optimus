from datasets import list_datasets, load_dataset
import tokenizers.normalizers as tn
import tokenizers.pre_tokenizers as tp
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers import Tokenizer

wiki = load_dataset('wikitext', 'wikitext-2-raw-v1')

with open('wiki.raw', 'wt') as f:
    for l in wiki['train']['text']:
        f.write(l)


vocab_size = 10_000

model = Unigram()



tokenizer = Tokenizer(model)
tokenizer.normalizer = tn.Sequence([
    tn.NFD(),
    tn.StripAccents(),
    tn.Lowercase(),
    tn.Strip()
])
tokenizer.pre_tokenizer = tp.Whitespace()
trainer = UnigramTrainer(
    vocab_size=vocab_size,
    special_tokens=["[UNK]", "[CLS]", "[PAD]", "[MASK]"]    
)

tokenizer.train(trainer, ['wiki.raw'])
pass
