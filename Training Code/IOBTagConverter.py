import pandas as pd
import transformers
import torch
import re


def target_info_extractor(df):
    target_word_indices = []
    target_words = []
    labels = []
    empty_target_counter = 0
    for i in df['targets']:
        # Inserting None to target=[] elements in order to keep the length in 2500
        if not i:
            empty_target_counter += 1
            target_word_indices.append(None)
            target_words.append(None)
            labels.append(None)
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
            target_word_indices.append(temp_twi)
            target_words.append(temp_tw)
            labels.append(temp_labels)

    return target_word_indices, target_words




class IOBTagConverter:
    def __init__(self, text, target_indices, target_words):
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
        self.text = text
        self.target_indices = target_indices
        self.target_words = target_words
        # 0 -> I , 1 -> O, 2-> B
        self.TAGS = ["0", "1", "2"]
        self.max_len = 92
        assert len(text) == len(target_indices) == len(target_words)

        for i in range(0, len(self.text)):
            # print(self.text[i])
            self.text[i] = re.sub(r'[^\w\s]', '', self.text[i])
            self.text[i] = self.text[i].replace('-','')
            self.text[i] = self.text[i].lstrip()

            # If target word has one,more then one, None words
            # If there are multiple target words in one instance
            if target_words[i]:
                for ix in self.target_words[i]:
                    word_counter = 0
                    for ix_splitted in ix.split():
                        ix_splitted = ix_splitted.replace('.', '')
                        if len(ix.split()) == 1:
                            self.text[i] = self.text[i].replace(ix_splitted, self.TAGS[2])
                        # For a target word including more then 1 word
                        elif len(ix.split()) > 1:
                            if word_counter == 0:
                                self.text[i] = self.text[i].replace(ix_splitted, self.TAGS[2])
                                word_counter += 1
                            else:
                                self.text[i] = self.text[i].replace(ix_splitted, self.TAGS[0],1)
                for word in self.text[i].split():
                    word = word.replace('.', '')
                    if word != '2' and word != '0':
                        if word not in self.target_words[i]:
                            self.text[i] = self.text[i].replace(word, self.TAGS[1], 1)




            else:  # For the None convert it directly to O-TAG
                for word in self.text[i].split():
                    self.text[i] = self.text[i].replace(word, self.TAGS[1], 1)


    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        tags = self.text[item]
        tags = [int(i) for i in tags.split()]
        # Padding is here with 3 for the labels ( TAGS )
        tags += [3] * (92 - len(tags))

        return {
            'tags': torch.tensor(tags, dtype=torch.long),
            'target_words': self.target_words[item],
            'target_indices': self.target_indices[item],
        }


# Loading procedure is with split() because there are some empty spaces in between because of deletion of not-ascii
def load_tags(file_path):
    iob_file = open(file_path, "r")

    iob_lines = []
    for lines in iob_file:
        lines = lines.split()
        iob_lines.append(lines)

    return iob_lines


if __name__ == '__main__':
    laptops_train_df = pd.read_json('data/laptops_train.json')
    laptops_dev_df = pd.read_json('data/laptops_dev.json')


    target_word_indices, target_words = target_info_extractor(laptops_train_df)

    tagger = IOBTagConverter(laptops_train_df['text'], target_word_indices, target_words)

    print(tagger[0])
