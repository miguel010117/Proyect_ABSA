import re
from unidecode import unidecode


class ExtractFeauresByWindows:

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path

    def create_new_sentence(self, sentence, word_or_phrase, windows_size):
        """
           Creates a new sentence containing a given word or phrase within a specified window size.

           Args:
               sentence (str): The original sentence
               word_or_phrase (str): The word or phrase to be included in the new sentence
               windows_size (int): The window size used to create the new sentence

           Returns:
               str or None: The new sentence or None if the word or phrase is not found in the original sentence
           """
        sentence_list = sentence.lower().split()
        word_or_phrase_list = word_or_phrase.split()
        if len(word_or_phrase_list) > 1:
            indices = [i for i, x in enumerate(sentence_list) if
                       x == word_or_phrase_list[0] and sentence_list[
                                                       i:i + len(word_or_phrase_list)] == word_or_phrase_list]
        else:
            indices = [i for i, x in enumerate(sentence_list) if x == word_or_phrase]
        if len(indices) > 0:
            index = indices[0]
            left_words = sentence_list[max(0, index - windows_size):index]
            right_words = sentence_list[
                          index + len(word_or_phrase_list):min(len(sentence_list),
                                                               index + len(word_or_phrase_list) + windows_size)]
            return ' '.join(left_words + word_or_phrase_list + right_words)
        else:
            return None

    def clean_text(self, text):
        """
           Cleans the text by removing special characters and non-ASCII characters.

           Args:
               text (str): The original text

           Returns:
               str: The cleaned text
           """

        text = re.sub(r'[^\w\s]', ' ', text)
        text = unidecode(text)
        return text

# def process_data(self, new_file_to_save_name, window_size):
#         """
#             Processes the data by creating new sentences and saving the processed data in a new csv file.

#             Args:
#                 new_file_to_save_name (str): The name of the new csv file to save the processed data
#                 window_size (int): The size of the windows to extract the new sentences.
#             Returns:
#                 None. The processed data is saved in a new csv file.
#         """
#         data = pd.read_csv(self.data_file_path, sep=';')
#         data = data[data['polarity'] != 'conflict']
#         data['reviews'] = data['reviews'].apply(self.clean_text)
#         data.dropna(inplace=True)

#         new_sentences = []
#         polarities = []
#         for i in range(len(data)):
#             sentence = data.iloc[i]['reviews']
#             word_or_phrase = data.iloc[i]['aspect']
#             new_sentence = self.create_new_sentence(sentence, word_or_phrase, windows_size=window_size)
#             polarity = data.iloc[i]['polarity']
#             if new_sentence is not None:
#                 new_sentences.append(new_sentence)
#                 polarities.append(polarity)

#         new_data = pd.DataFrame({'new_sentences': new_sentences, 'polarity': polarities})
#         new_data.drop_duplicates(inplace=True)
#         new_data.to_csv(new_file_to_save_name, index=False)

