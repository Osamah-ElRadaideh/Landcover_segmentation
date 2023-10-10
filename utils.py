def collate(example):
    # transforms a list of dictionaries to a dictionary of lists, sharing the same keys
    temp_dict = dict()
    for item in example:
        for key in item:
            if key not in temp_dict.keys():
                temp_dict[key] = []
            temp_dict[key].append(item[key])
    example = temp_dict
    return example
