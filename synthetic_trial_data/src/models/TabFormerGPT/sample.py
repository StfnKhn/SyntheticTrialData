import logging
from tqdm import tqdm
import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader

from synthetic_trial_data.src.preprocessing.data_sequencing import convert_sequences_to_tabular
from synthetic_trial_data.src.models.TabFormerGPT.datacollator import TransDataCollatorForLanguageModeling


logger = logging.getLogger(__name__)


class ClinicalSequenceGenerator:
    """
    A class to generate sequences using a trained TabFormerGPT model.

    :param tab_net: The trained TabFormerGPT model.
    :param dataset: The dataset containing vocabulary and related functions.
    :param train_dataset: The dataset for training.
    :param decoding: The method used for decoding. Defaults to "other".
    :param N_events: The number of events that should be predicted. Defaults to 3.
    :param temperature: The temperature parameter for softmax sampling. Defaults to 0.5.
    :param batch_size: The batch size used for the DataLoader. Defaults to 8.
    """

    def __init__(
        self, 
        tab_net,
        dataset,
        decoding="other",
        N_events: int = 3,
        temperature: float = 0.5,
        batch_size: int = 8,
        unique_id_col: str = None
        ):
        self.tab_net = tab_net
        self.dataset = dataset
        self.decoding = decoding
        self.N_events = N_events
        self.temperature = temperature
        self.unique_id_col = unique_id_col

        self.data_collator = TransDataCollatorForLanguageModeling(tokenizer=tab_net.tokenizer, mlm=False)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.data_collator.collate_batch)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tab_net.model.to(self.device)

        self.N_static = len(dataset.static_columns)
        self.N_event = len(dataset.event_columns)
        self.N_BOS_SEP = 2  # ["[BOS]", "[START_EVENT]"]

        # Initialize counter variable.
        self.event_columns_processed = 0

    def _insert_separator_token(self, generator_input):
        """Insert a separator token into the generator input."""
        separator_token_id = self.dataset.vocab.get_id(token="[START_EVENT]", field_name="SPECIAL", special_token=True)
        separator_token_tensor = torch.full((generator_input.size(0), 1), separator_token_id, dtype=torch.long).to(self.device)
        return torch.cat((generator_input, separator_token_tensor), dim=1)

    def generate_sequences(self, output_as_df=True):
        """
        Generate sequences using the TabFormerGPT model.

        :return: A list of generated sequences.
        :rtype: List[List[str]]
        """
        generated_samples = []
        dl_iter = iter(self.data_loader)

        for _ in tqdm(range(len(self.data_loader)), desc="Batch"):
            inputs = next(dl_iter)
            generator_input = inputs['labels'][:, :self.N_BOS_SEP + self.N_static + self.N_event].to(self.device)
            
            for i in range(self.N_events * self.N_event + self.N_events):
                if self.event_columns_processed != 0 and self.event_columns_processed % self.N_event == 0 or i == 0:
                    generator_input = self._insert_separator_token(generator_input)
                    self.event_columns_processed = 0
                    continue

                field_name = self.dataset.event_columns[self.event_columns_processed]
                global_ids_field = self.dataset.vocab.get_field_ids(field_name)
                generator_output = self.tab_net.model(generator_input)[0]
                lm_logits_field = generator_output[:, -1, global_ids_field]
        
                if self.decoding == 'greedy':
                    next_field_local_id = torch.max(lm_logits_field, dim=1)[1]
                else:
                    softmax_distribution = Categorical(logits=lm_logits_field / self.temperature)
                    # softmax decoding of field level local id
                    next_field_local_id = softmax_distribution.sample()
        
                token_id_to_add = self.dataset.vocab.get_from_local_ids(field_name=field_name, local_ids=next_field_local_id)
                generator_input = torch.cat((generator_input, token_id_to_add.unsqueeze(1)), dim=1)
                generator_input_tokens = self.dataset.vocab.get_from_global_ids(generator_input, 'tokens', with_field_name=False)

                # Increment the counter
                self.event_columns_processed += 1

            # Extend generated samples list with results from current batch
            generated_samples.extend(generator_input_tokens)

        if output_as_df:
            df = convert_sequences_to_tabular(
                sequences=generated_samples, 
                static_columns=self.dataset.static_columns,
                event_columns=self.dataset.event_columns, 
                sep_token=self.dataset.sep_token,
                unique_id_col=self.unique_id_col
            )
            return df
        else:
            return generated_samples
