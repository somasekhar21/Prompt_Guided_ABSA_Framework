from pydantic import BaseModel, Field
from typing import List, Dict, Literal
from langchain_core.output_parsers import PydanticOutputParser


class KeywordList(BaseModel):
    keywords: List[str] = Field(description = "List of keywords" ,default_factory=list)

    def __add__(self, other: "KeywordList") -> "KeywordList":
        return KeywordList(keywords=self.keywords + other.keywords)

class AspectKeywords(BaseModel):
    aspectKeywords: Dict[str,list[str]] = Field(description = "dictonary contains aspect catogeries as keys and associated keywords list as values " ,default_factory=dict)

class PrimaryOutput(BaseModel):
    target: str = Field(description = "target word" ,default="NULL")
    category: str = Field(description = "aspect category" ,default="NULL")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description = "sentiment of target,aspect")

class AspectSentiment(BaseModel):
    aspectSentiments: List[PrimaryOutput] = Field(description = "list of targets,aspect_catogeries,sentiments" ,default_factory=list)

keywordsParser = PydanticOutputParser(pydantic_object=KeywordList)
aspectKeywordsParser = PydanticOutputParser(pydantic_object=AspectKeywords)
aspectSentimentsParser = PydanticOutputParser(pydantic_object=AspectSentiment)

if __name__ == "__main__":
    print('output parsers')