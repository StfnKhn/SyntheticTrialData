"""
Copyright (c) 2021, International Business Machines (IBM). All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
"""

from transformers.tokenization_utils import PreTrainedTokenizer

class TabFormerTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    ):

        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token)