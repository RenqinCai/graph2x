import torch
import torchtext
from torchtext import data


class Vocab():
    def __init__(
            self, user_vocab, item_vocab, text_vocab,
            train_data, valid_data, test_data, text_field):
        # user/item/text vocab
        self.m_uservocab = user_vocab
        self.m_itemvocab = item_vocab
        self.m_textvocab = text_vocab
        # train/valid/test data
        # self.m_train_data = train_data
        self.m_valid_data = valid_data
        self.m_test_data = test_data
        # text field
        self.m_text_field = text_field
        # get special tokens and corresponding ids
        self.get_text_special_tokens()
        self.get_statistics()

    def get_text_special_tokens(self):
        """ Based on the text field object, we can get the special tokens
        from the text, including <unk>, <pad>, <sos> and <eos>
        """
        self.unk_token = self.m_text_field.unk_token
        self.pad_token = self.m_text_field.pad_token
        self.sos_token = self.m_text_field.init_token
        self.eos_token = self.m_text_field.eos_token
        self.unk_token_id = self.m_textvocab.stoi[self.unk_token]
        self.pad_token_id = self.m_textvocab.stoi[self.pad_token]
        self.sos_token_id = self.m_textvocab.stoi[self.sos_token]
        self.eos_token_id = self.m_textvocab.stoi[self.eos_token]

    def get_statistics(self):
        self.user_num = len(self.m_uservocab.itos)      # including <unk>
        self.item_num = len(self.m_itemvocab.itos)      # including <unk>
        self.vocab_size = len(self.m_textvocab.itos)    # this contains <unk>, <pad>, <sos>, <eos>
