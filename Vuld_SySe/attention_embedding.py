from torch import nn


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.ff = nn.Linear(in_features=hidden_dim, out_features=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, contexts, context_masks):
        """
        :param contexts: (batch_size, seq_len, n_hid)
        :param context_masks: (batch_size, seq_len)
        :return: (batch_size, n_hid)
        """
        out = self.ff(contexts)
        out = out.view(contexts.size(0), contexts.size(1))
        masked_out = out.masked_fill(context_masks, float('-inf'))
        attn_weights = self.softmax(masked_out)
        out = attn_weights.unsqueeze(1).bmm(contexts)
        out = out.squeeze(1)
        return out, attn_weights


class AttentionEmbedding(nn.Module):
    def __init__(self, emb_dim=256, hidden_dim=256, output_dim=2, external_token_embed=True, vocab_size=-1):
        super(AttentionEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.external_token_embed = external_token_embed
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if not self.external_token_embed:
            assert self.vocab_size != -1, 'Please provide vocabulary size to use embedding layer.'
            self.emb_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_dim)
        self.emb_transform = nn.Linear(in_features=self.emb_dim, out_features=self.hidden_dim)
        self.emb_drop = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(self.hidden_dim, 4, 2*self.hidden_dim, 0.1, 'relu')
        encoder_norm = nn.LayerNorm(self.hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, 6, encoder_norm)

        self.combiner = Attention(hidden_dim)
        self.output_layer = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, sequence, sequence_masks=None):
        """
        :param sequence: (batch_size, seq_len, hidden_dim) is external_token_embed == True else (batch_size, seq_len)
        :param sequence_masks: (batch_size, seq_len)
        :return: (batch_size, n_hid)
        """
        if self.external_token_embed:
            assert len(sequence.size()) == 3, 'Must provide a 3 dimension (batch_size * seq_len * hidden_dim) ' \
                                              'input for using external embedding'
            embedding = sequence.transpose(*(0, 1))
        else:
            assert len(sequence.size()) == 2, 'Must provide a 2 dimension (batch_size * seq_len) ' \
                                              'input for using external embedding'
            embedding = self.emb_layer(sequence)
            embedding = embedding.transpose(0, 1)
        transformed_embedding = self.emb_transform(embedding)
        encoded_embedding = self.encoder(transformed_embedding, src_key_padding_mask=sequence_masks)
        encoded_embedding = encoded_embedding.transpose(*(0, 1))
        combined_encoding, attn_weights = self.combiner(encoded_embedding, sequence_masks)
        output = self.output_layer(combined_encoding)
        return output, combined_encoding, attn_weights

