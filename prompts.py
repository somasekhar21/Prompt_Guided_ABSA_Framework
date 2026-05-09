from langchain_core.prompts import PromptTemplate
from outputparsers import keywordsParser,aspectKeywordsParser,aspectSentimentsParser

keywordsPrompt = PromptTemplate(
    template="""Instruction:
You are given:
1. A sentence : {sentence}
2. A list of allowed tokens keywords_list: {keywords}

Task:
Select tokens from the `keywords_list`  ONLY that appear EXACTLY in the sentence.
Tokens may include:
- Targets (nouns, entities)
- Opinion expressors (verbs, adjectives)
- Opinion intensifiers or modifiers
- Emoji strings in the format :emoji_name: which MUST be treated as sentiment or tone carriers

Rules (STRICT):
- Do NOT invent, infer, or rename tokens.
- Match tokens exactly as strings.
- Emoji strings (e.g., :family:) are valid tokens and must be selected if present.
- Output MUST NOT be empty if any keyword appears in the sentence.

{format_instructions}

""",
    input_variables=["sentence", "keywords"],
    partial_variables={"format_instructions": keywordsParser.get_format_instructions()})



aspectKeywordsMappingPrompt = PromptTemplate(
    template="""
Task Definition:
Match each keyword to the most relevant predefined aspect category based on the sentence context.
Select keywords not just by their type (noun, verb) but also by **their semantic or sentiment relevance** to the aspect.

ASPECT_CATEGORIES = {categories}

Instructions:
1. You are given a sentence: '{sentence}'
2. You are given a list of keywords: {keywords}
3. Assign the keyword to the most contextually relevant aspect.
4. If a keyword does not belong to any aspect, it may be assigned to "unknown" categorie.

{format_instructions}
""",
    input_variables=["sentence", "keywords","categories"],
    partial_variables={"format_instructions": aspectKeywordsParser.get_format_instructions()})




aspectSentimentPrompt=PromptTemplate(
    template="""Input
Sentence: {sentence}
Aspect_keywords = {aspect_keywords_str}

Task definition
Analyze the following sentence: "{sentence}"
Identify the target, aspect, and sentiment for each aspect found.
Known aspects and their related keywords are: {aspect_keywords_str}

Instruction
- Use the provided keywords and sentence context to identify the aspect.
- If a target is **not explicitly mentioned** but can be inferred from the context, treat the target as **NULL**.
- Use both the keywords and the sentence context to infer the sentiment (positive, negative, or neutral) for each aspect.

{format_instructions}
""",
    input_variables=["sentence", "aspect_keywords_str"],
    partial_variables={"format_instructions": aspectSentimentsParser.get_format_instructions()})


unknown_aspect_prompt=PromptTemplate(
    template="""Represent the aspect '{aspect}' as it appears in the sentence: '{sentence}' for clustering purpose. Focus on its specific meaning in this context.""",
    input_variables=["sentence", "aspect"]
)


if __name__ == "__main__":
    print('prompts')
