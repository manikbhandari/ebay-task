import torch.nn as nn
from transformers import BertForSequenceClassification
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sklearn_name2model = {
    'nb': MultinomialNB,
    'rf': RandomForestClassifier,
    'gb': GradientBoostingClassifier,
    'lr': LogisticRegression,
    'svm': svm.SVC
}


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name, num_labels=9)
        for n, p in self.encoder.named_parameters():
            p.requires_grad = False
        self.encoder.classifier.weight.requires_grad = True
        self.encoder.classifier.bias.requires_grad = True

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea

