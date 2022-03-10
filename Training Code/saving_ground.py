import torch
from itertools import groupby
from operator import itemgetter
import numpy as np


# 0 -> In
# 1 -> Out
# 2 -> Begin

def consecutive_groups(iterable, ordering=lambda x: x):
    for k, g in groupby(enumerate(iterable), key=lambda x: x[0] - ordering(x[1])):
        yield map(itemgetter(1), g)


# TODO: Return combinations of 2 and 0's.

# For the labels
def tensor_to_BIO(tensor):
    tensor = tensor.long().cpu().numpy()
    tensor = tensor[0]

    tags_to_convert = []

    padding_indices = np.where(tensor == 3)[0]
    tensor = np.delete(tensor, padding_indices)

    begin_indices = [i for i, x in enumerate(tensor) if x == 2]
    inside_indices = [i for i, x in enumerate(tensor) if x == 0]

    # print("Begin indices: ", begin_indices)
    # print("Inside indices: ", inside_indices)

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


    # print("Begin indices: ", begin_indices)
    # print("Inside indices: ", inside_indices)

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


if __name__ == '__main__':
    # c1,i = tensor_to_BIO(raw_tensor)
    # c2 = tensor_to_BIO_predictions(raw_tensor2,i)
    # print(c2)

    pass




