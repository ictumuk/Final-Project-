#------------------------------------------EVALUATION PROMPT----------------------------------------
EVAL_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

#------------------------------------------REWRITE THE PROMPT FOR WEBSEARCH----------------------------------------
WEBSEARCH_PROMPT = """Your task is to refine a query to ensure it is highly effective for retrieving relevant search results. \n
    Analyze the given input to grasp the core semantic intent or meaning. \n
    Original Query:
    \n ------- \n
    {question}
    \n ------- \n
    Your goal is to rephrase or enhance this query to improve its web search performance. Ensure the revised query is concise and directly aligned with the intended web search objective. \n
    Respond with the optimized query only:"""

#------------------------------------------REWRITE THE PROMPT FOR RETRIEVE----------------------------------------
RETRIEVE_PROMPT = """Your task is to refine the query to ensure it is highly effective in retrieving relevant results. \n
Analyze the input to understand the core semantic meaning or purpose.\n
Source query: {query}
List of old queries:
{old_queries}
Your goal is to rephrase or improve the query to enhance the retrieval performance of related texts.\n 
Make sure the revised query is concise and directly aligned with the retrieval purpose. Do not generate the old query. \n
Respond with the optimized query only:"""

#------------------------------------------TRUE/FALSE PROMPT----------------------------------------
TF_PROMPT = """
You are an assistant tasked with answering legal questions in Vietnamese. 
Use the following context to give a BINARY VALUE 'Đúng' or 'Sai' to answer the question. You must answer exactly in the given format 'Đúng' or 'Sai'.
An example Question: "The Earth is flat" Answer: "Sai", Question: "Honey never spoils" Answer: "Đúng"
Question: {question} 
Context: {context} 
Answer:
"""

#------------------------------------------MULTI-CHOICE PROMPT----------------------------------------
MC_PROMPT = """
You are an assistant tasked with answering legal multiple-choice questions in Vietnamese.\n 
Use the following context to give A SINGLE CHARACTER is 'A', 'B', 'C', or 'D'.\n.
Question: {question} 
Context: {context} 
Answer:
"""

#------------------------------------------FREE-TEXT PROMPT----------------------------------------
FT_PROMPT = """
You are an assistant tasked with answering legal free-text questions in Vietnamese. Use the following context to give a brief answer. 
An example,'Who is the 47th President of the United States?' Just answer 'Donald Trump', no need to provide answers like 'The President of the United States is Donald Trump' or any other unnecessary responses.\n
Question: {question}
Context: {context} 
Answer:
"""
#------------------------------------------REACT PROMPT----------------------------------------
REACT_PROMPT = '''Answer the following questions as best you can. 
Prioritize the use of retrieval tools. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question and use following format:

If the question type is Đúng/Sai, output should be a binary value [Đúng, Sai].
If the question type is Trắc nghiệm, output should be one of the four options [A, B, C, D].
If the question type is Tự luận, output should be concise, focusing on the main points without restating the question.

IMPORTANT: Every <Thought:> must either come with an <Action: and Action Input:> or <Final Answer:>

Begin!
Question Type: {question_type}
Question: {question}
Thought:{agent_scratchpad}'''