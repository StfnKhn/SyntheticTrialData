"""
Copyright (c) 2021, International Business Machines (IBM). All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

This implementation modifies the forward pass
"""

from torch.nn import CrossEntropyLoss

from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput


class TabFormerGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, vocab):
        super().__init__(config)
        self.vocab = vocab

    def forward(
            self,
            input_ids=None,
            #past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_cache=True,
            **kwargs
    ):
        transformer_outputs = self.transformer(
            input_ids,
            #past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        # lm_logits : [bsz x seq_len x vsz]
        # labels    : [bsz x seq_len]
        # When flatten is set to True:
        # seq_len = num_transactions * (num_columns + 2)  --> plus 2 because each transaction has BOS and EOS padding

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_labels = labels[..., 1:-1].contiguous()  # Remove first and last label: [BOS] and [EOS] tokens
            shift_logits = lm_logits[..., :-2, :].contiguous()  # Line up logits accordingly

            seq_len = shift_logits.size(1)
            total_lm_loss = 0
            field_names = self.vocab.get_field_keys(remove_target=True, ignore_special=True, ignore_static=True)

            N_static = len(self.vocab.static_columns)

            for field_idx, field_name in enumerate(field_names):
                col_ids = list(range(field_idx + N_static + 1, seq_len, len(field_names) + 1)) # +1 corresponds to the sep_token column
                global_ids_field = self.vocab.get_field_ids(field_name)
                lm_logits_field = shift_logits[:, col_ids, :][:, :, global_ids_field]  # bsz * 10 * K
                lm_labels_field = shift_labels[:, col_ids]
                
                lm_labels_local_field = self.vocab.get_from_global_ids(
                    global_ids=lm_labels_field,
                    what_to_get='local_ids'
                )

                #print("col_ids:", col_ids)
                #print("shift_labels:", shift_labels)
                #print("global_ids_field:", global_ids_field)
                #print("lm_labels_field:", lm_labels_field)
                #print("global_ids_field:", global_ids_field)
                #print("lm_labels_local_field", lm_labels_local_field)

                loss_fct = CrossEntropyLoss()
                lm_loss_field = loss_fct(lm_logits_field.view(-1, len(global_ids_field)),
                                         lm_labels_local_field.view(-1))
                total_lm_loss += lm_loss_field
                assert lm_labels_local_field.max() < len(global_ids_field), f"Max label {lm_labels_local_field.max()} is out of bounds!"

            outputs = (total_lm_loss,) + outputs

            return (total_lm_loss, BaseModelOutput(
                last_hidden_state=lm_logits,
                hidden_states=outputs[-2],
                attentions=outputs[-1]
            ))

            #return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
        else:
            # return BaseModelOutput(
            #     last_hidden_state=lm_logits,
            #     hidden_states=outputs[-2],
            #     attentions=outputs[-1]
            # )
            return CausalLMOutput(
                logits = lm_logits,
                hidden_states=outputs[-2],
                attentions=outputs[-1]
            )