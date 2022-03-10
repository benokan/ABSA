import string
import numpy as np
import pandas as pd
import transformers
from torch.utils import data
import re
from IOBTagConverter import IOBTagConverter
import torch
import copy


def target_info_extractor(df):
    target_word_indices = []
    target_words = []
    labels_second_part = []
    empty_target_counter = 0
    for c, i in enumerate(df['targets']):
        # Inserting None to target=[] elements in order to keep the length in 2500
        if not i:
            empty_target_counter += 1
            target_word_indices.append(None)
            target_words.append(None)
            labels_second_part.append(None)
        # If target is not empty
        else:
            # Create temporary lists to append to the real lists.
            # This is done for finding multiple aspects,target_indices in the targets.
            temp_twi = []
            temp_tw = []
            temp_labels = []
            # Looping through the length of the each target.
            for x in range(0, len(i)):
                temp_twi.append(i[x][0])
                temp_tw.append(i[x][1])
                temp_labels.append(i[x][2])
            # Sort by ascending index values
            srted = sorted(zip(temp_twi, temp_tw, temp_labels), key=lambda z: z[0][0])
            temp_twi = [k[0] for k in srted]
            temp_tw = [k[1] for k in srted]
            temp_labels = [k[2] for k in srted]
            target_word_indices.append(temp_twi)
            target_words.append(temp_tw)
            labels_second_part.append(temp_labels)

    assert len(target_word_indices) == len(target_words) == len(labels_second_part)
    return target_word_indices, target_words, labels_second_part


tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')



def check_max_length(dataframe):
    token_lengths = []
    for txt in dataframe['text']:
        tokens = tokenizer.encode(txt, max_length=5000)
        token_lengths.append(len(tokens))

    return max(token_lengths)


class LRABSADataset(data.Dataset):
    def __init__(self, text, target_indices, target_words, tokenizer, labels_second_part, second_part=False):
        self.text = text
        self.target_indices = target_indices
        self.target_words = target_words
        self.tokenizer = tokenizer
        self.second_part = second_part
        self.labels = list()
        # Raw data from extractor func
        self.labels_second_part = labels_second_part

        # If 0 it's the first one in the sequence, if [] there are no target words, if n (>0) n.th word same as
        # target word is the actual target word in the sequence
        self.target_word_extracted_indices = []
        self.TAGS = ["0", "1", "2"]

        self.second_part_inputs = []

        # Put them here after adapting it for the second part
        self.second_part_labels = []
        self.second_part_masks = []
        self.second_part_targets = []


        # Encoding target words
        for x, i in enumerate(self.target_words):
            temp_tw_list = []
            if i:
                for y in i:
                    temp_tw_list.append(tokenizer.encode(y, add_special_tokens=False))
            else:
                self.target_words[x] = []

            self.target_words[x] = temp_tw_list

        # Getting target words' indexes and which one is giving the exact position of the target word (Target_word_extracted_indices)
        for x, (i, ti) in enumerate(zip(self.text, self.target_indices)):
            if ti:
                tmp = []
                for target_indices in reversed(ti):
                    #               Until Target word                       Target Word                                               After Target word
                    self.text[x] = self.text[x][:target_indices[0]] + " " + self.text[x][
                                                                            target_indices[0]:target_indices[1]] + " " + \
                                   self.text[x][target_indices[1]:]
                    indices = [i for i in range(len(self.text[x])) if
                               self.text[x].startswith(self.text[x][target_indices[0] + 1:target_indices[1] + 1], i)]
                    tmp.append(indices.index(target_indices[0] + 1))
                self.target_word_extracted_indices.append(list(reversed(tmp)))

                self.text[x] = tokenizer.encode(self.text[x])
            else:
                self.text[x] = tokenizer.encode(self.text[x])
                self.target_word_extracted_indices.append([])

        # self.labels is a deep copy of self.text so from the beginning it will start converting text sequences into
        # tag sequences Converting labels to tags

        def find_target(sentence, target_word_list, occurence_indices):
            output = []
            for target_word, oc_in in zip(target_word_list, occurence_indices):
                _indices = []


                start = 0
                try:
                    while True:
                        word_indices = []
                        for word in target_word:
                            start = sentence.index(word, start)

                            word_indices.append(start)
                        start += len(target_word)

                        _indices.append(word_indices)

                except Exception as ex:
                    # print(ex)
                    pass

                output.append(_indices[oc_in])

            return output

        self.tidx = []

        faulty_indices = []
        for i in range(0, len(self.text)):
            if target_words[i]:
                try:
                    tidx = find_target(self.text[i], self.target_words[i], self.target_word_extracted_indices[i])
                    self.tidx.append(tidx)
                except Exception as e:
                    faulty_indices.append(i)
                    self.tidx.append(faulty_indices)
                    continue

                # Out Tag is 1
                labels = len(self.text[i]) * [1]
                for l in tidx:
                    for index, l1 in enumerate(l):
                        if index == 0:
                            labels[l1] = 2
                        else:
                            labels[l1] = 0

                self.labels.append(labels)


            else:
                labels = len(self.text[i]) * [1]
                self.labels.append(labels)
                self.tidx.append([])

        for i in reversed(faulty_indices):
            del self.text[i]
            del self.target_words[i]
            del self.target_word_extracted_indices[i]
            # FOR THE SECOND PART
            del self.labels_second_part[i]
            del self.tidx[i]

        if self.second_part:

            longest_string = max([j for i in self.tidx for j in i], key=len)
            longest_length = len(longest_string) + 10
            for k, i in enumerate(self.tidx):
                sec_sentence = self.text[k]

                for c, q in enumerate(i):
                    self.second_part_inputs.append(sec_sentence)
                    q += [-1] * (longest_length - len(q))
                    self.second_part_targets.append(q)

                    tmp = ['positive', 'negative', 'neutral', 'conflict']
                    target_label = self.labels_second_part[k][c]
                    target_label = tmp.index(target_label)
                    self.second_part_labels.append(target_label)


    def __len__(self):
        return len(self.text) if not self.second_part else len(self.second_part_inputs)

    def __getitem__(self, item):
        if not self.second_part:
            text_encodings = self.text[item]
            padded_mask = [1] * len(text_encodings)
            self.labels[item] += [3] * (128 - len(text_encodings))
            text_encodings += [0] * (128 - len(text_encodings))
            padded_mask += [0] * (128 - len(padded_mask))

            return {
                'input_ids': torch.tensor(text_encodings),
                'attention_mask': torch.tensor(padded_mask),
                'labels': torch.as_tensor(self.labels[item]),

            }
        else:
            second_text_encodings = self.second_part_inputs[item]
            padded_mask_2 = [1] * len(second_text_encodings)
            second_text_encodings += [0] * (100 - len(second_text_encodings))
            padded_mask_2 += [0] * (100 - len(padded_mask_2))

            return {
                'input_ids': torch.tensor(second_text_encodings),
                'attention_mask': torch.tensor(padded_mask_2),
                'second_part_labels': torch.as_tensor(self.second_part_labels[item]),
                'targets': torch.as_tensor(self.second_part_targets[item])

            }


if __name__ == '__main__':
    laptops_train_df = pd.read_json('C:/Users/beno/PycharmProjects/nlp2021-hw2-master/data/laptops_train.json')
    laptops_dev_df = pd.read_json('C:/Users/beno/PycharmProjects/nlp2021-hw2-master/data/laptops_dev.json')



    target_word_indices, target_words, labels_second_part = target_info_extractor(laptops_dev_df)



    ds = LRABSADataset(laptops_dev_df['text'].tolist(), target_word_indices, target_words, tokenizer,
                  labels_second_part=labels_second_part, second_part=True)

    loader = data.DataLoader(ds, batch_size=3)
    pass
