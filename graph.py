from langgraph.graph import StateGraph,END,START
from typing import TypedDict,Annotated,List
from prompts import *
from outputparsers import *
from categories import *
import os
import emoji
import operator
from utilityfunctions import emojiNormalise, posTagging, nlp, tokenizer, stop_words
from dotenv import load_dotenv
from modelsetup import load_llm_model,load_embedding_model
from vectorstoresetup import load_vectorstore_with_model
from outputparsers import aspectKeywordsParser,aspectSentimentsParser,keywordsParser

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    


load_dotenv()


model = load_llm_model()
embedding_model = load_embedding_model()
vectorstore=load_vectorstore_with_model()


class ABSAState(TypedDict,total=False):
    id:str
    sentence:str
    keywords:Annotated[KeywordList,operator.add]
    llmSelectionKeywords:Annotated[List[str],operator.add]
    aspect_keywords_mapping:AspectKeywords
    aspect_sentiments:AspectSentiment
    unknown_ids:Annotated[List[str], operator.add]



workflow = StateGraph(ABSAState)

#define nodes

def preprocess_sentence(state:ABSAState):

    keywordslist=[]
    sentence=state["sentence"]
    tokens=tokenizer.tokenize(sentence)

    #print(f"tokens: {tokens}\n")

    emojiList=emojiNormalise(tokens)

    #print(f"emojiList: {emojiList}\n")

    nonEmojiTokens=[i for i in tokens if not emoji.is_emoji(i)]

    #print(f"nonEmojiTokens: {nonEmojiTokens}\n")

    posTaggedKeywords,llmSelectionKeywords=posTagging(nonEmojiTokens)

    #print(f"posTaggedKeywords: {posTaggedKeywords}\n")
    #print(f"llmSelectionKeywords: {llmSelectionKeywords}\n")

    keywordslist=emojiList+posTaggedKeywords

    print(f"preprocessing node done.....")


    return {"keywords": KeywordList(keywords=keywordslist), "llmSelectionKeywords": llmSelectionKeywords}
   
    
def llm_keywords_node(state:ABSAState):

    if state['llmSelectionKeywords']:
        llmSelectionKeywords=state["llmSelectionKeywords"]
        prompt = keywordsPrompt.invoke({"sentence":state["sentence"],"keywords":llmSelectionKeywords})
        response = model.invoke(prompt)
        #print(f"response: {response.content}")
        keywordsobj = keywordsParser.parse(response.content)

    print(f"llm keywords node done.....")

    return {"keywords": keywordsobj}
    

def llm_aspect_keywords_mapping_node(state:ABSAState):
    
    keywords=state["keywords"].keywords
    prompt = aspectKeywordsMappingPrompt.invoke({"sentence":state["sentence"],"keywords":keywords,"categories":ASPECT_CATEGORIES})
    response = model.invoke(prompt)
    #print(f"response: {response.content}")
    aspectKeywordsobj = aspectKeywordsParser.parse(response.content)

    print(f"llm aspect keywords mapping node done.....")

    return {"aspect_keywords_mapping": aspectKeywordsobj}
    

def llm_aspect_sentiments_node(state:ABSAState):
    
    aspectKeywordsobj=state["aspect_keywords_mapping"]
    aspect_keywords_str = ",".join([f"{aspect}:{','.join(keywords)}" for aspect, keywords in aspectKeywordsobj.aspectKeywords.items()])
    prompt = aspectSentimentPrompt.invoke({"sentence":state["sentence"],"aspect_keywords_str":aspect_keywords_str})
    response = model.invoke(prompt)
    #print(f"response: {response.content}")
    aspectSentimentobj = aspectSentimentsParser.parse(response.content)

    print(f"llm aspect sentiments node done.....")

    return {"aspect_sentiments": aspectSentimentobj}

def unknown_handle_node(state:ABSAState):

    aspect_keywords_dict=state["aspect_keywords_mapping"].aspectKeywords
    unknownSentiments=[]
    ids=[]
    if "unknown" in aspect_keywords_dict.keys():
        keywords=aspect_keywords_dict.get("unknown",[])
        for i in state['aspect_sentiments'].aspectSentiments:
            if i.category=="unknown":
                unknownSentiments.append(i.model_dump_json())

        sentence=state["sentence"]
        prompt=unknown_aspect_prompt.format(sentence=sentence,aspect=keywords)
        vectorstore.add_texts(texts=[prompt],metadatas=[{"aspect_keywords":keywords,"sentence":sentence,"sentiments":unknownSentiments,"id":state["id"]}],ids=[state["id"]])
        ids.append(state["id"])

        print(f"unknown content in sentences ids {ids}\n")

    print(f"unknown handle node done.....")

    return {'unknown_ids':ids}

#add nodes
workflow.add_node("preprocess_sentence",preprocess_sentence)
workflow.add_node("llm_keywords_node",llm_keywords_node)
workflow.add_node("llm_aspect_keywords_mapping_node",llm_aspect_keywords_mapping_node)
workflow.add_node("llm_aspect_sentiments_node",llm_aspect_sentiments_node)
workflow.add_node("unknown_handle_node",unknown_handle_node)


# add edges
workflow.add_edge(START,"preprocess_sentence")
workflow.add_edge("preprocess_sentence","llm_keywords_node")
workflow.add_edge("llm_keywords_node","llm_aspect_keywords_mapping_node")
workflow.add_edge("llm_aspect_keywords_mapping_node","llm_aspect_sentiments_node")
workflow.add_edge("llm_aspect_sentiments_node","unknown_handle_node")
workflow.add_edge("unknown_handle_node",END)


#build
workflow = workflow.compile()

if __name__ == "__main__":
    print("Graph built successfully\n")