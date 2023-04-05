import torch

import open_clip

from training.imagenet_zeroshot_data import (
    imagenet_classnames,
    openai_imagenet_template,
)

zero_class = ''

embeds_tens = None

model, _, _ = open_clip.create_model_and_transforms(
    'ViT-bigG-14', pretrained='laion2b_s39b_b160k')
model = model.to('cuda', dtype=torch.bfloat16)
tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

# text = tokenizer(["a diagram", "a dog", "a cat"])

embeds_tens = torch.empty((1001, 42, 77, 1280))

for i, cn in enumerate(imagenet_classnames):
    print('Current class:', i, cn)
    texts = [fn(cn) for fn in openai_imagenet_template]
    texts = tokenizer(texts).to('cuda')
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text_unsquished(texts)
        tf_on_cpu = text_features.float().cpu()
        embeds_tens[i] = tf_on_cpu

with torch.no_grad(), torch.cuda.amp.autocast():
    texts = tokenizer([zero_class]).to('cuda')
    text_features = model.encode_text_unsquished(texts)
    embeds_tens[1000] = text_features.float().cpu().tile(
        (len(openai_imagenet_template),1,1))

print('Final shape:', embeds_tens.shape)
torch.save(embeds_tens, 'imagenet_embeddings.pt')
