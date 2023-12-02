from transformers.modeling_utils import PreTrainedModel
from transformers import (
    GPT2Config,
    GPT2LMHeadModel
)

from synthetic_trial_data.src.models.TabFormerGPT.tabformer_tokenizer import TabFormerTokenizer
from synthetic_trial_data.src.models.TabFormerGPT.hierarchical import TabFormerEmbeddings
from synthetic_trial_data.src.models.TabFormerGPT.tabformer_gpt2 import TabFormerGPT2LMHeadModel
from synthetic_trial_data.src.models.TabFormerGPT.utils import ddict



class TabFormerBaseModel(PreTrainedModel):
    def __init__(self, hf_model, tab_embeddings, config):
        super().__init__(config)

        self.model = hf_model
        self.tab_embeddings = tab_embeddings

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerGPT2:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False):

        self.vocab = vocab
        self.config = GPT2Config(vocab_size=len(self.vocab))

        self.tokenizer = TabFormerTokenizer(
            unk_token=special_tokens.unk_token,
            bos_token=special_tokens.bos_token,
            eos_token=special_tokens.eos_token
        )

        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):
        if field_ce:
            model = TabFormerGPT2LMHeadModel(self.config, self.vocab)
        else:
            model = GPT2LMHeadModel(self.config)
        if not flatten:
            tab_emb_config = ddict(vocab_size=len(self.vocab), hidden_size=self.config.hidden_size)
            model = TabFormerBaseModel(model, TabFormerEmbeddings(tab_emb_config))

        return model