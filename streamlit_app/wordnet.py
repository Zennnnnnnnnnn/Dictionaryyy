import nltk
# Ensure you have the necessary nltk data
nltk.download('wordnet')
from nltk.corpus import wordnet
# Functions for WordNet
def create_english_dictionary(word):
        synsets = wordnet.synsets(word)
        english_dictionary = []

        for index, synset in enumerate(synsets, start=1):
            synset_id = synset.name()
            definition = synset.definition()
            synonyms = set(lemma.name() for lemma in synset.lemmas())
            antonyms = set(lemma.name() for lemma in synset.lemmas() if lemma.antonyms())
            parents = [str(parent.name()) for parent in synset.hypernyms()]
            children = [str(child.name()) for child in synset.hyponyms()]
            examples = synset.examples()

            pos = synset.pos()
            pos_mapping = {
                'n': 'noun',
                'v': 'verb',
                'a': 'adjective',
                'r': 'adverb'
            }
            part_of_speech = pos_mapping.get(pos, 'unknown')

            entry = {
                'Synset_ID': synset_id,
                'Description': definition,
                'Synonyms': list(synonyms),
                'Antonyms': list(antonyms),
                'Parents': parents,
                'Children': children,
                'Examples': examples,
                'Word_Type': part_of_speech
            }

            english_dictionary.append(entry)

        return english_dictionary
