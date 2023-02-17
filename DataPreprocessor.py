import os


def get_list_of_images(path_to_dataset):
    return os.listdir(path_to_dataset)


def get_all_unique_chars(list_of_images):
    image_ns = [image_fn.split(".")[0] for image_fn in list_of_images]
    image_ns = "".join(image_ns)
    return sorted(list(set(list(image_ns))))


def add_to_letters(letters, char='-'):
    return [char] + letters


def map_char_to_integer(letters):
    return {k: v for k, v in enumerate(letters, start=0)}


def map_integer_to_char(idx2char):
    return {v: k for k, v in idx2char.items()}


def get_maps_from_path(path_to_dataset):
    letters = get_all_unique_chars(get_list_of_images(path_to_dataset))
    letters = add_to_letters(letters, char='-')

    idx2char = map_char_to_integer(letters)
    char2idx = map_integer_to_char(idx2char)
    return idx2char, char2idx
