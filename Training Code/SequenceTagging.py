import pandas as pd
import transformers
import torch
from torch import nn
from torch.utils import data
from preprocessing import target_info_extractor
from preprocessing import LRABSADataset
from IOBTagConverter import IOBTagConverter
import pickle
import pytorch_lightning as pl
from transformers import BertModel
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score, \
    classification_report
from saving_ground import tensor_to_BIO, tensor_to_BIO_predictions
import numpy as np
from TorchCRF import CRF
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import json

pickle_switch = False


class DiceLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = 3

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ignore_condition = torch.ne(targets, self.ignore_index)
        logits = logits[ignore_condition]
        targets = targets[ignore_condition]

        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")


def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def load_from_pickle(ds):
    file_to_read = open(ds, "rb")
    loaded_instances = pickle.load(file_to_read)
    file_to_read.close()
    global pickle_switch
    pickle_switch = True
    return loaded_instances


def save_pickle(ds, i):
    file_to_store = open(f"stored_object{i}.pickle", "wb")
    pickle.dump(ds, file_to_store)
    file_to_store.close()


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


    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels_duplicate = labels.detach().clone()

        logits = self(input_ids, attention_mask, True)

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
        print("val acc:", val_acc)
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
        self.class_weights = torch.FloatTensor(self.normalized_weights).cuda()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.config = self.bert.config.to_dict()
        with open("model1_sequenceTaggingNoCrf_config.json", 'w') as f:
            json.dump(self.config, f)

        self.criterion = nn.CrossEntropyLoss(ignore_index=3, weight=self.class_weights)
        self.hidden = nn.Linear(768, 768)
        self.fc = nn.Linear(768, 3)
        self.drop = nn.Dropout(p=0.3)
        self.diceLoss = DiceLoss()

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

        # f1 = f1_score(labels.view(-1).cpu(), predictions.view(-1).cpu(), average='weighted')

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

        # f1 = f1_score(labels.view(-1).cpu(), predictions.view(-1).cpu(), average='weighted')
        ld = labels_duplicate.view(-1).cpu().tolist()
        pd = predictions.view(-1).cpu().tolist()


        class_report = classification_report(ld, pd, target_names=['Inside', 'Outside', 'Begin'])
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
class BertASC(pl.LightningModule):
    def __init__(self, normalized_weights):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.config = self.bert.config.to_dict()
        with open("config.json", 'w') as f:
            json.dump(self.config, f)

        self.normalized_weights = torch.FloatTensor(normalized_weights).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=self.normalized_weights)
        self.hidden = nn.Linear(768, 768)
        self.fc = nn.Linear(768, 4)
        self.drop = nn.Dropout(p=0.7)

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
        print("Those are the predictions: ", predictions)
        return {"val_loss": loss, "val_acc": val_acc, "class_report": class_report, 'macro_f1': macro_f1,
                'positive_f1': positive_f1, 'negative_f1': negative_f1, 'neutral_f1': neutral_f1,
                'conflict_f1': conflict_f1}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # this calls forward
        return self(batch)

    def validation_epoch_end(self, outputs):
        macro_f1 = torch.tensor([x['macro_f1'] for x in outputs]).mean()
        positive_f1 = torch.tensor([x['positive_f1'] for x in outputs]).mean()
        negative_f1 = torch.tensor([x['negative_f1'] for x in outputs]).mean()
        neutral_f1 = torch.tensor([x['neutral_f1'] for x in outputs]).mean()
        conflict_f1 = torch.tensor([x['conflict_f1'] for x in outputs]).mean()
        val_acc = torch.tensor([x['val_acc'] for x in outputs]).mean()
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        print("val_acc ", val_acc)
        print("macro f1 ", macro_f1)
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


if __name__ == '__main__':


    print("Didn't load the object from pickle file.")
    laptops_train_df = pd.read_json('data/laptops_train.json')
    laptops_dev_df = pd.read_json('data/laptops_dev.json')

    # restaurants_train_df = pd.read_json('data/restaurants_train.json')
    # restaurants_dev_df = pd.read_json('data/restaurants_dev.json')

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')

    target_word_indices, target_words, labels_second_part_train = target_info_extractor(laptops_train_df)
    target_word_indices_dev, target_words_dev, labels_second_part_dev = target_info_extractor(laptops_dev_df)

    # ds_task_a = LRABSADataset(laptops_train_df['text'].tolist(), target_word_indices, target_words, tokenizer,
    #                           labels_second_part=labels_second_part_train, second_part=False)
    # ds_dev_task_a = LRABSADataset(laptops_dev_df['text'].tolist(), target_word_indices_dev, target_words_dev, tokenizer,
    #                               labels_second_part=labels_second_part_dev, second_part=False)

    ds_task_b = LRABSADataset(laptops_train_df['text'].tolist(), target_word_indices, target_words, tokenizer,
                             labels_second_part=labels_second_part_train, second_part=True)

    ds_dev_task_b = LRABSADataset(laptops_dev_df['text'].tolist(), target_word_indices_dev, target_words_dev, tokenizer,
                               labels_second_part=labels_second_part_dev, second_part=True)

    train_loader = data.DataLoader(
        ds_task_b,
        batch_size=32,
        num_workers=4,
        shuffle=True,
    )

    dev_loader = data.DataLoader(
        ds_dev_task_b,
        batch_size=32,
        num_workers=4,
        shuffle=False,
    )

    # FIRST PART WEIGHTING
    # count_I = 0
    # count_O = 0
    # count_B = 0
    # for i in train_loader:
    #     i['labels'] = i['labels'].numpy()
    #     count_I += np.count_nonzero(i['labels'] == 0)
    #     count_O += np.count_nonzero(i['labels'] == 1)
    #     count_B += np.count_nonzero(i['labels'] == 2)
    #
    # class_weights_IOB = [count_I, count_O, count_B]
    # normedWeights = [1 - (x / sum(class_weights_IOB)) for x in class_weights_IOB]
    #
    # print(normedWeights)
    # exit()

    #
    # model_crf = BertCRFSequenceTagger(normalized_weights=normedWeights, lstm=False)
    # model = BertSequenceTagger(normalized_weights=normedWeights, lstm=False)




    # SECOND PART WEIGHTING
    count_positive = 0
    count_neg = 0
    count_neut = 0
    count_conflict = 0
    for i in train_loader:
        i['second_part_labels'] = i['second_part_labels'].numpy()
        count_positive += np.count_nonzero(i['second_part_labels'] == 0)
        count_neg += np.count_nonzero(i['second_part_labels'] == 1)
        count_neut += np.count_nonzero(i['second_part_labels'] == 2)
        count_conflict += np.count_nonzero(i['second_part_labels'] == 3)

    normedWeights_sec = [count_positive, count_neg, count_neut, count_conflict]
    normedWeights = [1 - (x / sum(normedWeights_sec)) for x in normedWeights_sec]
    print(normedWeights)

    model = BertASC(normalized_weights=normedWeights)

    # model = model.load_from_checkpoint(
    #     checkpoint_path='lightning_logs/version_4/checkpoints/epoch=4-step=394.ckpt',
    #     normalized_weights=normedWeights,
    # )
    #


    model.load_state_dict(torch.load('models/merged_model_classification_50f1.pth',map_location='cpu'))




    early_stop_callback = EarlyStopping(
        monitor='macro_f1',
        min_delta=0.00,
        patience=6,
        verbose=False,
        mode='max'
    )

    model_checkpoint_callback = ModelCheckpoint(
        monitor='macro_f1',
        mode='max'
    )

    trainer = pl.Trainer(max_epochs=1, gpus=1, fast_dev_run=False,
                         callbacks=[early_stop_callback,
                                    model_checkpoint_callback
                                    ])

    # model.eval()
    # trainer.validate(model, dev_loader)

    trainer.fit(model, train_loader, dev_loader)

    torch.save(model.state_dict(), "models/merged_model_classification_50f1_fineTunedForLaptops.pth")
