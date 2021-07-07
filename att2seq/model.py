import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import numpy as np
import random
import time


class Encoder(nn.Module):
    def __init__(self, user_num, item_num, enc_hid_dim, dec_hid_dim, rnn_layers, rating_range):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.rnn_layers = rnn_layers
        self.user_embedding = nn.Embedding(user_num, self.enc_hid_dim)
        self.item_embedding = nn.Embedding(item_num, self.enc_hid_dim)
        self.rating_embedding = nn.Embedding(rating_range + 1, self.enc_hid_dim)
        self.hidden_layer = nn.Linear(self.enc_hid_dim * 3, self.dec_hid_dim * self.rnn_layers)

    def forward(self, user, item, rating):
        # user/item/rating shape: (batch_size)
        # embed shape: (batch_size, enc_hid_dim)
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        rating_embed = self.rating_embedding(rating)
        # concat embed shape: (batch_size, enc_hid_dim * 3)
        concat_embed = torch.cat((user_embed, item_embed, rating_embed), dim=-1)
        # hidden_state shape: (batch_size, dec_hid_dim, rnn_layers)
        hidden_state = torch.tanh(self.hidden_layer(concat_embed)).view(-1, self.dec_hid_dim, self.rnn_layers)

        '''TEST REVIEW GENERATION'''
        # # # # START ------ ****** verify review generation with GRU ****** ####
        # Instead of generate a initial hidden state based on features, use a all-zero placeholder to test
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # hidden_state = torch.zeros(1, user.shape[0], config.hidden_dim).to(device)
        # ****** verify review generation with GRU ****** ------ END ####

        return hidden_state, user_embed, item_embed, rating_embed


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout):
        super().__init__()
        # ouptut_dim: vocabulary size of the review text
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hid_dim = dec_hid_dim
        # word embedding layer
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # Decoder RNN
        self.rnn = nn.GRU(emb_dim, dec_hid_dim, num_layers=self.num_layers, dropout=dropout)
        # self.attention_fc = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        # self.decodetop_fc = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        # Word output (without attention)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        # dropout for word embedding
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input shape: (batch_size)
        # hidden shape: (n_layers*n_directions, batch_size, dec_hid_dim)
        # encoder_outputs shape: (feature_length(=3), batch_size, enc_hid_dim)
        input = input.unsqueeze(0)                              # input shape: (1, batch_size)
        word_embed = self.dropout(self.embedding(input))        # word_embed shape: (1, batch_size, word_embed_dim)

        # Compute decoder hidden state
        # hidden: (n_layers*n_directions, batch_size, dec_hid_dim)
        # cell: (n_layers*n_directions, batch_size, dec_hid_dim)
        # output: (seq_len(=1), batch_size, num_directions*dec_hid_dim)
        output, hidden = self.rnn(word_embed, hidden)

        # NOTE: seq_len and num_directions will always be 1 in the decoder, thus:
        # hidden: (n_layers, batch_size, dec_hid_dim)
        # cell: (n_layers, batch_size, dec_hid_dim)
        # output: (1, batch_size, dec_hid_dim)
        prediction = self.fc_out(output.squeeze(0))             # prediction shape: (batch_size, output_dim)
        return prediction, hidden


class Att2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, tf_ratio, max_seq_length):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = tf_ratio
        self.max_seq_length = max_seq_length

    def set_tf_ratio(self, tf_ratio):
        self.teacher_forcing_ratio = tf_ratio

    def forward(self, user, item, rating, text):
        # user/item/rating: (batch_size)
        # text: (text_len, batch_size)
        # teacher_forcing_ratio is probability to use teacher forcing
        batch_size = user.shape[0]
        text_length = text.shape[0]
        text_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(text_length, batch_size, text_vocab_size).to(self.device)
        start_time = time.time()
        hidden, user_embed, item_embed, rating_embed = self.encoder(user, item, rating)
        end_time = time.time()
        encoder_time = end_time - start_time

        # construct initial hidden state
        # (encoder output) hidden: (batch_size, dec_hid_dim, rnn_layers)
        hidden = hidden.permute(2, 0, 1).contiguous()
        # (after permute) hidden: (rnn_layers, batch_size, dec_hid_dim)

        # first input to the decoder is the <sos> tokens
        input = text[0, :]

        # TODO: change this token-by-token teacher forcing to full batch input
        start_time = time.time()
        for t in range(1, text_length):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.teacher_forcing_ratio

            # get the highest predicted token from the prediction
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = text[t] if teacher_force else top1
        end_time = time.time()
        decoder_total_time = end_time - start_time
        print("Encoder compute time: {0} \t Decoder compute time: {1} (text length: {2})".format(
            encoder_time, decoder_total_time, text_length
            )
        )

        return outputs

    def eval_forward(self, user, item, rating, text):
        """ teacher_forcing ratio should be 0
        """
        # user/item/rating: (batch_size)
        # text: (text_len, batch_size)
        # teacher_forcing_ratio is probability to use teacher forcing
        batch_size = user.shape[0]
        text_length = self.max_seq_length + 2       # add <sos> and <eos> tokens
        text_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(text_length, batch_size, text_vocab_size).to(self.device)
        hidden, user_embed, item_embed, rating_embed = self.encoder(user, item, rating)

        # construct initial hidden state
        # (encoder output) hidden: (batch_size, dec_hid_dim, rnn_layers)
        hidden = hidden.permute(2, 0, 1).contiguous()
        # (after permute) hidden: (rnn_layers, batch_size, dec_hid_dim)

        # first input to the decoder is the <sos> tokens
        input = text[0, :]

        # TODO: optimize this output generation. when all data instances in this batch has meet
        # <eos>, the generation should stop.

        for t in range(1, text_length):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # get the highest predicted token from the prediction
            top1 = output.argmax(1)

            # in eval we don't have teacher forcing, then should use predicted token
            input = top1

        return outputs
