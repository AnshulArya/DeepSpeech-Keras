import argparse
import numpy as np
import pandas as pd
from collections import Counter

from source.utils import chdir, create_logger
from source.text import Alphabet
from source.metric import edit_distance, naive_backtrace, decode_
from plots import save_confusion_matrix, save_donut


def update_(confusion_matrix: np.ndarray, to_substitute: dict, alphabet: Alphabet):
    """ Update the confusion matrix. """
    for correct_char, wrong_chars in to_substitute.items():
        correct_char_label = alphabet.label_from_string(correct_char)
        wrong_chars_labels = [alphabet.label_from_string(char) for char in wrong_chars]

        for wrong_char_label in wrong_chars_labels:
            confusion_matrix[correct_char_label, wrong_char_label] += 1


def main(alphabet_path: str, results_path: str, home_directory: str):
    alphabet = Alphabet(alphabet_path)
    inserts, deletes = Counter(), Counter()
    confusion_matrix = np.zeros([alphabet.size, alphabet.size], dtype=int)
    results = pd.read_csv(results_path, usecols=['original', 'prediction'])

    for index, original, prediction in results.itertuples():
        distance, edit_distance_matrix, backtrace = edit_distance(source=prediction,
                                                                  destination=original)
        best_path = naive_backtrace(backtrace)
        to_delete, to_insert, to_substitute = decode_(best_path, prediction, original)
        update_(confusion_matrix, to_substitute, alphabet)
        inserts.update(to_insert)
        deletes.update(to_delete)

    save_confusion_matrix(confusion_matrix, labels=alphabet._label_to_str, directory=home_directory)
    save_donut(inserts, deletes, confusion_matrix, directory=home_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphabet', required=True, help='Alphabet as the txt file')
    parser.add_argument('--results', required=True, help='The csv file contains `original` and `prediction` columns')
    parser.add_argument('--home_directory', required=True, help='Directory where save all files')
    parser.add_argument('--log_file', help='Log file')
    parser.add_argument('--log_level', type=int, default=20, help='Log level')
    arguments = parser.parse_args()
    chdir(to='ROOT')

    logger = create_logger(arguments.log_file, level=arguments.log_level, name='confusion_matrix')
    logger.info(f'Arguments: \n{arguments}')
    main(arguments.alphabet, arguments.results, arguments.home_directory)
