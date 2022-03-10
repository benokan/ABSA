import numpy as np
from typing import List, Tuple, Dict

from model import Model
import random
import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertModel, BertConfig
import pandas as pd
import os
import logging
from torch.utils import data
import transformers
from TorchCRF import CRF
import json
from stud.preprocessing import target_info_extractor,LRABSADataset
from itertools import groupby
from operator import itemgetter
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score, \
    classification_report




def consecutive_groups(iterable, ordering=lambda x: x):
    for k, g in groupby(enumerate(iterable), key=lambda x: x[0] - ordering(x[1])):
        yield map(itemgetter(1), g)


def tensor_to_BIO(tensor):
    tensor = tensor.long().cpu().numpy()
    tensor = tensor[0]

    tags_to_convert = []

    padding_indices = np.where(tensor == 3)[0]
    tensor = np.delete(tensor, padding_indices)

    begin_indices = [i for i, x in enumerate(tensor) if x == 2]
    inside_indices = [i for i, x in enumerate(tensor) if x == 0]


    def recursive_populate(search_index, list_to_search):
        if search_index in list_to_search:
            tags_to_convert.append(search_index)
            return recursive_populate(search_index + 1, list_to_search)
        else:
            pass


    # TODO: Populate a new list with B-BI-BII-BIIIIIIIIIIIIIII stuff.

    for begins in begin_indices:
        recursive_populate(begins + 1, inside_indices)

    consecutive_tags = [list(group) for group in consecutive_groups(tags_to_convert)]


    combined_tags = []
    for x, indices in enumerate(begin_indices):
        temp = []

        try:
            for consec_tag in consecutive_tags:
                if indices + 1 in consec_tag:
                    temp.append(indices)
                    temp.extend(consec_tag)
                    combined_tags.append(temp)
            else:
                if not temp:
                    combined_tags.append([indices])
        except IndexError:
            print("index out of bounds")

    combined_tags = [tuple(i) for i in combined_tags]


    return combined_tags, padding_indices

# For the predictions
def tensor_to_BIO_predictions(tensor, to_delete):
    tensor = tensor.long().cpu().numpy()
    tensor = tensor[0]


    tags_to_convert = []

    padding_indices = to_delete
    tensor = np.delete(tensor, padding_indices)




    begin_indices = [i for i, x in enumerate(tensor) if x == 2]
    inside_indices = [i for i, x in enumerate(tensor) if x == 0]


    def recursive_populate(search_index, list_to_search):
        if search_index in list_to_search:
            tags_to_convert.append(search_index)
            return recursive_populate(search_index + 1, list_to_search)
        else:
            pass

    # We have all the indices of the tags now.
    # TODO: Populate a new list with B-BI-BII-BIIIIIIIIIIIIIII stuff.

    for begins in begin_indices:
        recursive_populate(begins + 1, inside_indices)

    consecutive_tags = [list(group) for group in consecutive_groups(tags_to_convert)]
    # print("Consecutive tags: ", consecutive_tags)

    combined_tags = []
    for x, indices in enumerate(begin_indices):
        temp = []

        try:
            for consec_tag in consecutive_tags:
                if indices + 1 in consec_tag:
                    temp.append(indices)
                    temp.extend(consec_tag)
                    combined_tags.append(temp)
            else:
                if not temp:
                    combined_tags.append([indices])
        except IndexError:
            print("index out of bounds")

    combined_tags = [tuple(i) for i in combined_tags]


    return combined_tags

######################
def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

# Part A
class BertCRFSequenceTagger(pl.LightningModule):
    def __init__(self, normalized_weights, lstm=False):
        super().__init__()
        # For labels [0,1,2] -> [I,O,B] Since we have tons of O. I'll try giving more weight to B & I
        self.normalized_weights = normalized_weights
        self.class_weights = torch.FloatTensor(self.normalized_weights).cuda()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.criterion = nn.CrossEntropyLoss(ignore_index=3, weight=self.class_weights)
        self.hidden = nn.Linear(768, 768)
        self.fc = nn.Linear(768, 3)
        self.drop = nn.Dropout(p=0.3)
        self.crf = CRF(3, batch_first=True)

    def forward(self, input_ids, attention_mask, training):
        output = self.bert(input_ids, attention_mask=attention_mask)
        # Dropout only in training phase
        if training:
            output = self.hidden(output[0])
            output = torch.relu(output)
            output = self.drop(output)
            logits = self.fc(output)
            return logits

        else:
            logits = self.hidden(output[0])
            logits = torch.relu(logits)
            logits = self.fc(logits)
            return logits

        # output = torch.softmax(logits, dim=-1)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels_duplicate = labels.detach().clone()

        logits = self(input_ids, attention_mask, True)

        # Converting the padding indices to out tags for CRF
        labels_3_indices = (labels == 3).nonzero(as_tuple=True)
        for b, s in zip(labels_3_indices[0], labels_3_indices[1]):
            labels[b][s] = 1

        loss_crf = -self.crf(logits, labels, attention_mask.byte())

        outputs = torch.softmax(logits, dim=-1)
        predictions = outputs.argmax(dim=-1)

        # TODO: Remove padding indexes from labels for metrics calculations.
        # TODO: And the corresponding indexes from predictions

        # This section is to calculate f1 seperately
        tp, fp, fn = 0, 0, 0

        for label, prediction in zip(labels, predictions):
            label_bio, pad_indices_to_delete = tensor_to_BIO(label.unsqueeze(0))
            prediction_bio = tensor_to_BIO_predictions(prediction.unsqueeze(0), pad_indices_to_delete)
            label_bio = set(label_bio)
            prediction_bio = set(prediction_bio)
            tp += len(prediction_bio & label_bio)
            fn += len(prediction_bio - label_bio)
            fp += len(label_bio - prediction_bio)

        precision, f1, recall = 0, 0, 0

        if not tp == 0:
            precision = 100 * tp / (tp + fp)
            recall = 100 * tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

        indices_to_delete_labels = (labels_duplicate.view(-1) == 3).nonzero(as_tuple=True)[0]
        labels_duplicate = th_delete(labels_duplicate.view(-1), indices_to_delete_labels)
        predictions = th_delete(predictions.view(-1), indices_to_delete_labels)

        assert labels_duplicate.shape == predictions.shape

        ld = labels_duplicate.view(-1).cpu().tolist()
        pd = predictions.view(-1).cpu().tolist()


        train_acc = balanced_accuracy_score(ld, pd)

        self.log("train_f1", f1)
        self.log("train_acc", train_acc, logger=True)
        self.log("train_loss", loss_crf, logger=True)
        return {"loss": loss_crf, "train_acc": train_acc}


    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels_duplicate = labels.detach().clone()

        logits = self(input_ids, attention_mask, False)

        # Converting the padding indices to out tags for CRF
        labels_3_indices = (labels == 3).nonzero(as_tuple=True)
        for b, s in zip(labels_3_indices[0], labels_3_indices[1]):
            labels[b][s] = 1

        loss_crf = -self.crf(logits, labels, attention_mask.byte())

        outputs = torch.softmax(logits, dim=-1)
        predictions = outputs.argmax(dim=-1)

        # TODO: Remove padding indexes from labels for metrics calculations.
        # TODO: And the corresponding indexes from predictions.
        # This section is to calculate f1 seperately
        tp, fp, fn = 0, 0, 0

        for label, prediction in zip(labels, predictions):
            label_bio, pad_indices_to_delete = tensor_to_BIO(label.unsqueeze(0))
            prediction_bio = tensor_to_BIO_predictions(prediction.unsqueeze(0), pad_indices_to_delete)

            label_bio = set(label_bio)
            prediction_bio = set(prediction_bio)

            tp += len(prediction_bio & label_bio)
            fp += len(prediction_bio - label_bio)
            fn += len(label_bio - prediction_bio)

        precision, f1, recall = 0, 0, 0

        if not tp == 0:
            precision = 100 * (tp / (tp + fp))
            recall = 100 * (tp / (tp + fn))
            f1 = 2 * precision * recall / (precision + recall)

        # Removing the padding indices
        indices_to_delete_labels = (labels_duplicate.view(-1) == 3).nonzero(as_tuple=True)[0]
        labels_duplicate = th_delete(labels_duplicate.view(-1), indices_to_delete_labels)
        predictions = th_delete(predictions.view(-1), indices_to_delete_labels)

        assert labels_duplicate.shape == predictions.shape

        ld = labels_duplicate.view(-1).cpu().tolist()
        pd = predictions.view(-1).cpu().tolist()


        val_acc = balanced_accuracy_score(ld, pd)
        class_report = classification_report(ld, pd, target_names=['Inside', 'Outside', 'Begin'])

        print("\n", class_report)

        self.log("f1_score", f1)
        self.log("val_acc", val_acc, logger=True, prog_bar=True)
        self.log("val_loss", loss_crf, logger=True)
        return {"val_loss": loss_crf, "val_acc": val_acc}
        # return {"val_loss": loss_crf}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.fc.parameters()},
             {'params': self.crf.parameters()},
             {'params': self.hidden.parameters()}]
            , lr=1e-3)

        return optimizer
class BertSequenceTagger(pl.LightningModule):
    def __init__(self, normalized_weights, lstm=False):
        super().__init__()
        # For labels [0,1,2] -> [I,O,B] Since we have tons of O. I'll try giving more weight to B & I
        self.normalized_weights = normalized_weights
        self.class_weights = torch.FloatTensor(self.normalized_weights)
        # self.bert = BertModel.from_pretrained('bert-base-cased')
        self.config = BertConfig.from_json_file('model/laptops_without_crf/checkpoints/model1_sequenceTaggingNoCrf_config.json')
        self.bert = BertModel(self.config)
        self.criterion = nn.CrossEntropyLoss(ignore_index=3, weight=self.class_weights)
        self.hidden = nn.Linear(768, 768)
        self.fc = nn.Linear(768, 3)
        self.drop = nn.Dropout(p=0.3)


    def forward(self, input_ids, attention_mask, training):
        output = self.bert(input_ids, attention_mask=attention_mask)
        # Dropout only in training phase
        if training:
            output = self.hidden(output[0])
            output = torch.relu(output)
            output = self.drop(output)
            logits = self.fc(output)
            return logits

        else:
            logits = self.hidden(output[0])
            logits = torch.relu(logits)
            logits = self.fc(logits)
            return logits


    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels_duplicate = labels.detach().clone()

        logits = self(input_ids, attention_mask, True)

        loss = self.criterion(logits.view(-1, 3), labels.view(-1))
        # loss = self.diceLoss(logits.view(-1, 3), labels.view(-1))

        outputs = torch.softmax(logits, dim=-1)
        predictions = outputs.argmax(dim=-1)

        # TODO: Remove padding indexes from labels for metrics calculations.
        # TODO: And the corresponding indexes from predictions
        # This section is to calculate f1 seperately
        tp, fp, fn = 0, 0, 0

        for label, prediction in zip(labels, predictions):
            label_bio, pad_indices_to_delete = tensor_to_BIO(label.unsqueeze(0))
            prediction_bio = tensor_to_BIO_predictions(prediction.unsqueeze(0), pad_indices_to_delete)
            label_bio = set(label_bio)
            prediction_bio = set(prediction_bio)
            tp += len(prediction_bio & label_bio)
            fn += len(prediction_bio - label_bio)
            fp += len(label_bio - prediction_bio)

        precision, f1, recall = 0, 0, 0

        if not tp == 0:
            precision = 100 * tp / (tp + fp)
            recall = 100 * tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

        indices_to_delete_labels = (labels_duplicate.view(-1) == 3).nonzero(as_tuple=True)[0]
        labels_duplicate = th_delete(labels_duplicate.view(-1), indices_to_delete_labels)
        predictions = th_delete(predictions.view(-1), indices_to_delete_labels)

        assert labels_duplicate.shape == predictions.shape

        ld = labels_duplicate.view(-1).cpu().tolist()
        pd = predictions.view(-1).cpu().tolist()


        train_acc = balanced_accuracy_score(ld, pd)

        self.log("train_f1", f1)
        self.log("train_acc", train_acc, logger=True)
        self.log("train_loss", loss, logger=True)
        return {"loss": loss, "train_acc": train_acc}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels_duplicate = labels.detach().clone()

        logits = self(input_ids, attention_mask, False)

        loss = self.criterion(logits.view(-1, 3), labels.view(-1))
        # loss = self.diceLoss(logits.view(-1, 3), labels.view(-1))

        outputs = torch.softmax(logits, dim=-1)
        predictions = outputs.argmax(dim=-1)

        # TODO: Remove padding indexes from labels for metrics calculations.
        # TODO: And the corresponding indexes from predictions.
        # This section is to calculate f1 seperately
        tp, fp, fn = 0, 0, 0
        for label, prediction in zip(labels, predictions):
            label_bio, pad_indices_to_delete = tensor_to_BIO(label.unsqueeze(0))
            prediction_bio = tensor_to_BIO_predictions(prediction.unsqueeze(0), pad_indices_to_delete)

            label_bio = set(label_bio)
            prediction_bio = set(prediction_bio)

            tp += len(prediction_bio & label_bio)
            fp += len(prediction_bio - label_bio)
            fn += len(label_bio - prediction_bio)

        precision, f1, recall = 0, 0, 0

        if not tp == 0:
            precision = 100 * (tp / (tp + fp))
            recall = 100 * (tp / (tp + fn))
            f1 = 2 * precision * recall / (precision + recall)

        # Removing the padding indices
        indices_to_delete_labels = (labels_duplicate.view(-1) == 3).nonzero(as_tuple=True)[0]
        labels_duplicate = th_delete(labels_duplicate.view(-1), indices_to_delete_labels)
        predictions = th_delete(predictions.view(-1), indices_to_delete_labels)

        assert labels_duplicate.shape == predictions.shape

        ld = labels_duplicate.view(-1).cpu().tolist()
        pd = predictions.view(-1).cpu().tolist()

        val_acc = balanced_accuracy_score(ld, pd)

        self.log("f1_score", f1)
        self.log("val_acc", val_acc, logger=True, prog_bar=True)
        self.log("val_loss", loss, logger=True)
        return {"val_loss": loss, "val_acc": val_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.fc.parameters()},
             {'params': self.hidden.parameters()}]
            , lr=1e-3)

        return optimizer


# Part B
class BertASC(pl.LightningModule):
    def __init__(self, normalized_weights):
        super().__init__()
        # self.bert = BertModel.from_pretrained('bert-base-cased')
        self.config = BertConfig.from_json_file('model/laptops_part_b/checkpoints/config.json')
        self.bert = BertModel(self.config)
        # Since it was generating an error removed the .cuda() line... Cuda is installed on WSL btw.
        # self.normalized_weights = torch.FloatTensor(normalized_weights).cuda()
        self.normalized_weights = torch.FloatTensor(normalized_weights)

        self.criterion = nn.CrossEntropyLoss(weight=self.normalized_weights)
        self.hidden = nn.Linear(768, 768)
        self.fc = nn.Linear(768, 4)
        self.drop = nn.Dropout(p=0.0)

    def forward(self, input_ids, target_words, attention_mask, training):
        output = self.bert(input_ids, attention_mask=attention_mask)

        # Data has been checked nothing seems peculiar
        logit_list = []

        for o, t in zip(output[0], target_words):
            padding_mask = torch.ne(t, -1)
            t = t[padding_mask]
            aggregated_target_word = torch.mean(o[t], dim=0) + torch.mean(o, dim=0)

            if training:
                output = self.drop(aggregated_target_word)
                output = self.hidden(output)
                output = torch.relu(output)
                output = self.drop(output)
                logits = self.fc(output)
                logit_list.append(logits)
            else:
                logits = self.hidden(aggregated_target_word)
                logits = torch.relu(logits)
                logits = self.fc(logits)
                logit_list.append(logits)

        logit_list = torch.stack(logit_list, dim=0)

        return logit_list

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        target_words_step = batch["targets"]
        attention_mask = batch["attention_mask"]
        labels = batch["second_part_labels"]

        logits = self(input_ids, target_words_step, attention_mask, True)
        loss = self.criterion(logits, labels)

        outputs = torch.softmax(logits, dim=-1)
        predictions = outputs.argmax(dim=-1)

        train_acc = balanced_accuracy_score(labels.view(-1).cpu().tolist(), predictions.view(-1).cpu().tolist())

        self.log("train_acc", train_acc, logger=True)
        self.log("train_loss", loss, logger=True)
        return {"loss": loss, "train_acc": train_acc}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_words_step = batch["targets"]
        labels = batch["second_part_labels"]

        logits = self(input_ids, target_words_step, attention_mask, False)
        loss = self.criterion(logits, labels)
        outputs = torch.softmax(logits, dim=-1)
        predictions = outputs.argmax(dim=-1)

        class_report = classification_report(labels.cpu().tolist(), predictions.cpu().tolist(), labels=[0, 1, 2, 3],
                                             target_names=['positive', 'negative', 'neutral', 'conflict'],
                                             output_dict=True)

        positive_f1, negative_f1, neutral_f1, conflict_f1 = list(), list(), list(), list()

        positive_f1.append(class_report['positive']['f1-score'])
        negative_f1.append(class_report['negative']['f1-score'])
        neutral_f1.append(class_report['neutral']['f1-score'])
        conflict_f1.append(class_report['conflict']['f1-score'])

        val_acc = balanced_accuracy_score(labels.view(-1).cpu().tolist(), predictions.view(-1).cpu().tolist())
        macro_f1 = f1_score(labels.cpu().tolist(), predictions.cpu().tolist(), average='macro')

        return {"val_loss": loss, "val_acc": val_acc, "class_report": class_report, 'macro_f1': macro_f1,
                'positive_f1': positive_f1, 'negative_f1': negative_f1, 'neutral_f1': neutral_f1,
                'conflict_f1': conflict_f1}

    def validation_epoch_end(self, outputs):
        macro_f1 = torch.tensor([x['macro_f1'] for x in outputs]).mean()
        positive_f1 = torch.tensor([x['positive_f1'] for x in outputs]).mean()
        negative_f1 = torch.tensor([x['negative_f1'] for x in outputs]).mean()
        neutral_f1 = torch.tensor([x['neutral_f1'] for x in outputs]).mean()
        conflict_f1 = torch.tensor([x['conflict_f1'] for x in outputs]).mean()
        val_acc = torch.tensor([x['val_acc'] for x in outputs]).mean()
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log("macro_f1", macro_f1, logger=True)
        self.log("positive_f1", positive_f1, logger=True)
        self.log("negative_f1", negative_f1, logger=True)
        self.log("neutral_f1", neutral_f1, logger=True)
        self.log("conflict_f1", conflict_f1, logger=True)
        self.log("val_acc", val_acc, logger=True)
        self.log("val_loss", val_loss, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.fc.parameters()},
             {'params': self.hidden.parameters()},
            # {'params': self.bert.parameters(), 'lr': 1e-5},
             ]
            , lr=1e-4)

        return optimizer

######################

######################

def build_model_b(device: str) -> Model:
    
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements aspect sentiment analysis of the ABSA pipeline.
            b: Aspect sentiment analysis.
    """
    return StudentModel(device,mode='b')

def build_model_ab(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline.
            a: Aspect identification.
            b: Aspect sentiment analysis.

    """
    return StudentModel(device,mode='ab')


def build_model_cd(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline 
        as well as Category identification and sentiment analysis.
            c: Category identification.
            d: Category sentiment analysis.
    """
    # return RandomBaseline(mode='cd')
    raise NotImplementedError

class RandomBaseline(Model):

    options_sent = [
        ('positive', 793+1794),
        ('negative', 701+638),
        ('neutral',  365+507),
        ('conflict', 39+72),
    ]

    options = [
        (0, 452),
        (1, 1597),
        (2, 821),
        (3, 524),
    ]

    options_cat_n = [
        (1, 2027),
        (2, 402),
        (3, 65),
        (4, 6),
    ]

    options_sent_cat = [
        ('positive', 1801),
        ('negative', 672),
        ('neutral',  411),
        ('conflict', 164),
    ]

    options_cat = [
        ("anecdotes/miscellaneous", 939),
        ("price", 268),
        ("food", 1008),
        ("ambience", 355),
    ]

    def __init__(self, mode = 'b'):

        self._options_sent = [option[0] for option in self.options_sent]
        self._weights_sent = np.array([option[1] for option in self.options_sent])
        self._weights_sent = self._weights_sent / self._weights_sent.sum()

        if mode == 'ab':
            self._options = [option[0] for option in self.options]
            self._weights = np.array([option[1] for option in self.options])
            self._weights = self._weights / self._weights.sum()
        elif mode == 'cd':
            self._options_cat_n = [option[0] for option in self.options_cat_n]
            self._weights_cat_n = np.array([option[1] for option in self.options_cat_n])
            self._weights_cat_n = self._weights_cat_n / self._weights_cat_n.sum()

            self._options_sent_cat = [option[0] for option in self.options_sent_cat]
            self._weights_sent_cat = np.array([option[1] for option in self.options_sent_cat])
            self._weights_sent_cat = self._weights_sent_cat / self._weights_sent_cat.sum()

            self._options_cat = [option[0] for option in self.options_cat]
            self._weights_cat = np.array([option[1] for option in self.options_cat])
            self._weights_cat = self._weights_cat / self._weights_cat.sum()

        self.mode = mode

    def predict(self, samples: List[Dict]) -> List[Dict]:

        preds = []
        for sample in samples:
            pred_sample = {}
            words = None
            if self.mode == 'ab':
                n_preds = np.random.choice(self._options, 1, p=self._weights)[0]
                if n_preds > 0 and len(sample["text"].split(" ")) > n_preds:
                    words = random.sample(sample["text"].split(" "), n_preds)
                elif n_preds > 0:
                    words = sample["text"].split(" ")
            elif self.mode == 'b':
                if len(sample["targets"]) > 0:
                    words = [word[1] for word in sample["targets"]]
            if words:
                pred_sample["targets"] = [(word, str(np.random.choice(self._options_sent, 1, p=self._weights_sent)[0])) for word in words]
            else: 
                pred_sample["targets"] = []
            if self.mode == 'cd':
                n_preds = np.random.choice(self._options_cat_n, 1, p=self._weights_cat_n)[0]
                pred_sample["categories"] = []
                for i in range(n_preds):
                    category = str(np.random.choice(self._options_cat, 1, p=self._weights_cat)[0]) 
                    sentiment = str(np.random.choice(self._options_sent_cat, 1, p=self._weights_sent_cat)[0]) 
                    pred_sample["categories"].append((category, sentiment))
            preds.append(pred_sample)
        return preds


class StudentModel(Model):
    def __init__(self,device,mode):
        self.device = device
        # STUDENT: construct here your model
        # this class should be loading your weights and vocabulary
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
        # Weights for loss function doesn't have much importance here but for testing I've written those from a run in local
        # Weights for part 1 (SequenceTagger)-> [0.9663628767416101, 0.07030566837022212, 0.9633314548881677]
        # Weights for part 2 (BertASC) ->[0.5824634655532359, 0.6304801670146138, 0.80741127348643, 0.9796450939457203]
        self.weights_1 = [0.9663628767416101, 0.07030566837022212, 0.9633314548881677]
        self.weights_2 = [0.5824634655532359, 0.6304801670146138, 0.80741127348643, 0.9796450939457203]
        self.model_BertASC = BertASC(normalized_weights=self.weights_2)
        self.model_BertASC.to(device)
        self.model_NoCRFTagger = BertSequenceTagger(normalized_weights=self.weights_1,lstm=False)
        self.model_NoCRFTagger.to(device)
        self.mode = mode



        self.model_BertASC.load_state_dict(torch.load('model/combined_models/merged_model_classification_50f1.pth',map_location='cpu'))
        logging.error("Loaded the model 2 succesfully")
        self.model_NoCRFTagger.load_state_dict(torch.load('model/combined_models/merged_model_sequence_tagging.pth',map_location='cpu'))
        logging.error("Loaded the model 1 succesfully")
        
    
    def grouper(self,list1,list2):
        groups = []
        for i in list1:
            word = [i]
            tmp = i
            for i2 in list2:
                if tmp+1 == i2:
                    tmp = i2
                    word.append(tmp)
            groups.append(word)
            
        return groups

    
    def predict_ab(self,samples):
        labels = ['positive','negative','neutral','conflict']

        predictions_a = []
        predictions_ab = []
        
        for sample in samples:
            tmp_predictions = {'targets':[]}
            encoded_sentence = self.tokenizer.encode(sample["text"],add_special_tokens=True)
            
            mask = [1]*len(encoded_sentence)
            encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
            encoded_sentence = encoded_sentence.to(self.device)
            mask = torch.tensor(mask).unsqueeze(0)
            mask = mask.to(self.device)
            logits_a = self.model_NoCRFTagger(encoded_sentence,mask,training=False)
            outputs_a = torch.softmax(logits_a, dim=-1)
            pred_a = torch.argmax(outputs_a, dim=-1).tolist()
            predictions_a.append(pred_a)

            begin_indices = [i for i, x in enumerate(pred_a[0]) if x == 2]
            inside_indices = [i for i, x in enumerate(pred_a[0]) if x == 0]


            grouped = self.grouper(begin_indices,inside_indices)

            
            for target in grouped:
                decoded = self.tokenizer.decode(encoded_sentence.squeeze()[target])
                target = torch.tensor(target).unsqueeze(0)
                encoded_sentence = encoded_sentence.to(self.device)
                target = target.to(self.device)
                mask = mask.to(self.device)
                logits_ab = self.model_BertASC(encoded_sentence,target,mask,training=False)
                outputs_ab = torch.softmax(logits_ab, dim=-1)
                pred_ab = torch.argmax(outputs_ab, dim=-1)
                tmp_predictions['targets'].append((decoded,labels[pred_ab]))

            predictions_ab.append(tmp_predictions)

        # logging.error(("predictions: ",predictions))

        return predictions_ab

    def predict_b(self,samples):           
        labels = ['positive','negative','neutral','conflict']
        predictions = []
        for sample in samples:
            tmp_predictions = {'targets':[]}
            sentence = sample["text"]
            
            for target in sample["targets"]:
                indices = target[0]
                target_words = target[1]
                part1 = self.tokenizer.encode(sentence[:indices[0]],add_special_tokens = False)
                part2 = self.tokenizer.encode(sentence[indices[0]:indices[1]],add_special_tokens = False)
                part3 = self.tokenizer.encode(sentence[indices[1]:],add_special_tokens = False)

                whole_sentence_tokenized = [101] + part1 + part2 + part3 + [102]
                target_indices = [len(part1)+i for i in range(len(part2))]
                mask = [1]*len(whole_sentence_tokenized)

                whole_sentence_tokenized = torch.tensor(whole_sentence_tokenized).unsqueeze(0)
                whole_sentence_tokenized = whole_sentence_tokenized.to(self.device)
                target_indices = torch.tensor(target_indices).unsqueeze(0)
                target_indices = target_indices.to(self.device)
                mask = torch.tensor(mask).unsqueeze(0)
                mask = mask.to(self.device)

                logits = self.model_BertASC(whole_sentence_tokenized,target_indices,mask,training=False)
                outputs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(outputs, dim=-1)

                tmp_predictions['targets'].append((target_words,labels[pred]))
            predictions.append(tmp_predictions)

        return predictions



    def predict(self, samples: List[Dict]) -> List[Dict]:
        # predictions_b = self.predict_b(samples)
        if self.mode == 'ab':
            predictions = self.predict_ab(samples)
            return predictions
        if self.mode == 'b':
            predictions_b = self.predict_b(samples)
            return predictions_b


