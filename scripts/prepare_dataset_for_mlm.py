from datasets import list_datasets, load_dataset, Dataset
from tokenizers import Tokenizer
import torch
from functools import partial

def transform_data(batch, tokenizer:Tokenizer):
    p = 0.2
    text = batch["text"]
    encoded = tokenizer.encode(text)

    orig_mask = torch.rand(len(encoded.ids)) < p
    allowed_mask = torch.logical_not(torch.BoolTensor(encoded.special_tokens_mask))
    final_mask = torch.logical_and(orig_mask, allowed_mask)

    ids = torch.LongTensor(encoded.ids)
    ids[final_mask] = tokenizer.token_to_id("[MASK]")
    special_tokens_mask = torch.LongTensor(encoded.special_tokens_mask)
    special_tokens_mask[final_mask] = 1

    return {
        "labels": encoded.ids,
        'special_tokens_mask':special_tokens_mask,
        "ids": ids,
    }

def prepare_tokenizer():
    tokenizer = Tokenizer.from_file("notebooks/tokenizer.json")
    tokenizer.enable_padding(
        pad_token="[PAD]", pad_id=tokenizer.token_to_id("[PAD]"), length=128
    )
    tokenizer.enable_truncation(max_length=128)

    tokenizer.save('tokenizer128.json', pretty=True)

    return tokenizer

def main():
    tokenizer = prepare_tokenizer()

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]
    print(wiki)
    wiki = wiki.filter(lambda x: len(x["text"]) > 0)

    dataset_transform = partial(transform_data, tokenizer=tokenizer)
    wiki:Dataset = wiki.map(dataset_transform, batched=False)
    wiki.set_format('torch', ['labels', 'special_tokens_mask', 'ids'])

    print(wiki)
    wiki.save_to_disk('wiki128')


if __name__ == '__main__':
    main()
