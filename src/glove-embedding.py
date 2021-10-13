
from flair.embeddings import WordEmbeddings
from flair.data import Sentence


def read_data(datafile):
    with open(datafile) as f:
        lines = f.readlines()
    return lines

def main():
    # init embedding
    glove_embedding = WordEmbeddings('glove')

    words = read_data('vertomul.txt')
    print(words[0])
  # create sentence.
    for word in words:
        print(word)
        sentence = Sentence(word.strip())
         # embed a sentence using glove.
        glove_embedding.embed(sentence)
        for token in sentence:
            print(token)
            print(token.embedding)

    


# main function
if __name__ == "__main__":
    main()