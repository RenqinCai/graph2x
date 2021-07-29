import torch
import torchtext
from torchtext import data


class Vocab():
    def __init__(
            self, text_vocab, text_field):
        # user/item/text vocab
        self.m_textvocab = text_vocab       # feature words
        # train/valid/test data
        # self.m_train_data = train_data
        # self.m_valid_data = valid_data
        # self.m_test_data = test_data
        # text field
        self.m_text_field = text_field      # feature words
        # get special tokens and corresponding ids
        self.get_text_special_tokens()
        self.get_statistics()

    def get_text_special_tokens(self):
        """ Based on the text field object, we can get the special tokens
        from the text, including <unk>, <pad>, <sos> and <eos>
        """
        self.unk_token = self.m_text_field.unk_token
        self.pad_token = self.m_text_field.pad_token
        self.unk_token_id = self.m_textvocab.stoi[self.unk_token]
        self.pad_token_id = self.m_textvocab.stoi[self.pad_token]

    def get_statistics(self):
        self.vocab_size = len(self.m_textvocab.itos)    # this contains <unk>, <pad>

    def set_usernum(self, user_num):
        self.user_num = user_num      # does not contain <unk>

    def set_itemnum(self, item_num):
        self.item_num = item_num      # does not contain <unk>
