import torch
import torch.nn as nn
from transformers import *
from torchcrf import CRF

class BertWithCRF(nn.Module):
    def __init__(self, num_classes):
        super(BertWithCRF, self).__init__()
        # self.bert_model = BertModel.from_pretrained('bert-base-chinese', hidden_dropout_prob=0.1,
        #                                             output_hidden_states=True)
        self.bert_model = BertModel.from_pretrained("bert-base-chinese", hidden_dropout_prob=0.1,
                                                    output_hidden_states=True)
        self.hidden_size = 768
        self.dropout = 0.3
        self.dropout = nn.Dropout(self.dropout)
        self.position_wise_ff = nn.Linear(self.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, ids, seg_ids, tags=None):
        attention_mask = (ids > 0)

        # outputs: (last_encoder_layer, pooled_output, attention_weight)
        outputs = self.bert_model(input_ids=ids,
                                token_type_ids=seg_ids,
                                attention_mask=attention_mask)

        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.position_wise_ff(last_encoder_layer)

        if tags is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, tags, reduction='none'), self.crf.decode(emissions)
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags
