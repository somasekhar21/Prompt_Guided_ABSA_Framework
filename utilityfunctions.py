import nltk
from nltk import TweetTokenizer
from nltk.corpus import stopwords
import emoji
import spacy



def emojiNormalise(keywordslist):
    
    emojis_to_demojize = [k for k in keywordslist if emoji.is_emoji(k)]
    demojizedList = [emoji.demojize(e) for e in emojis_to_demojize]
    print(f"emojis presented: {emojis_to_demojize} and its associated names: {demojizedList}")
    return demojizedList


def loadSpacyModel():
    model_name = "en_core_web_sm"
    try:
        nlp = spacy.load(model_name,disable=["parser", "ner", "senter"])
    except Exception as e:
        print(f"Failed to load SpaCy model: {e}")
        return "error-1"

    return nlp

def loadStopWords():
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
   
    return stop_words

def loadTokenizer():
    try:
        tokenizer = TweetTokenizer()
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return "error-2"
    return tokenizer

def posTagging(keywordslist):
    
    posTaggedKeywords=[]
    llmSelectionKeywords=[]
    docs=list(nlp.pipe(keywordslist))
    for doc in docs:
        token=doc[0]
        if token.lemma_.lower() not in stop_words:
            if token.pos_ in ["NOUN","ADJ"]:
                posTaggedKeywords.append(token.text)
            else:
                llmSelectionKeywords.append(token.text)
    return posTaggedKeywords,llmSelectionKeywords

nlp=loadSpacyModel()
tokenizer=loadTokenizer()
stop_words=loadStopWords()


def embed_documents():
    contextual_instruction_template = "Represent the aspect '{aspect}' as it appears in the sentence: '{sentence}' for clustering purpose. Focus on its specific meaning in this context."

    
    

if __name__ == "__main__":
    print("utility functions")