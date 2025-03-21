from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from .prompt import (EVAL_PROMPT,
                    TF_PROMPT,
                    MC_PROMPT,
                    FT_PROMPT,
                    WEBSEARCH_PROMPT,
                    RETRIEVE_PROMPT)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class Node:
    """
        A class to manage the Retrieval-Augmented Generation (RAG) workflow.

        Args:
            llm (BaseChatModel): The language model to use for generation and grading
            retrieve_tool: Tool for document retrieval
            web_search_tool: Tool for web searching
        """

    def __init__(
        self,
        llm,
        retrieve_tool,
        web_search_tool,
    ):
        self.llm = llm
        self.retrieve_tool = retrieve_tool
        self.web_search_tool = web_search_tool
        self._setup_prompts()
        self._setup_chains()


    def _setup_prompts(self):
        # Setup prompt templates with defaults or custom prompts
        # Prompt Question & Answer
        self.tf_prompt = PromptTemplate(template=TF_PROMPT)
        self.mc_prompt = PromptTemplate(template=MC_PROMPT)
        self.ft_prompt = PromptTemplate(template=FT_PROMPT)

        # Prompt Rewrite
        self.rewrite_retrieve_prompt = PromptTemplate(template=RETRIEVE_PROMPT)
        self.rewrite_web_prompt = PromptTemplate(template=WEBSEARCH_PROMPT)

        # Prompt Grade
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", EVAL_PROMPT),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

    def _setup_chains(self):

        """Setup LLM chains and structured output"""
        # Structured output model for document grading
        class GradeDocuments(BaseModel):
            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )

        # Create chains
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader

        # Query rewriting chains
        self.web_question_rewriter = self.rewrite_web_prompt | self.llm | StrOutputParser()
        self.retrieve_question_rewriter = self.rewrite_retrieve_prompt | self.llm | StrOutputParser()

        # RAG generation chains
        self.rag_chain_tf = self.tf_prompt | self.llm | StrOutputParser()
        self.rag_chain_mc = self.mc_prompt | self.llm | StrOutputParser()
        self.rag_chain_ft = self.ft_prompt | self.llm | StrOutputParser()

    def generate(self, state):
        """
        Generates a response based on the provided state that includes question,
        documents, and question type. Depending on the question type, it uses
        different RAG chains to generate the appropriate output. The method
        integrates different generation paths tailored for True/False, Multiple
        Choice, or Free-Text question types.

        :param state: A dictionary containing the following keys:
                      - question: str
                      - documents: list
                      - question_type: str
        :return: A dictionary containing the original documents, the question,
                 and the generated results based on the question type.
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        question_type = state["question_type"]
        # RAG generation
        if question_type == "Đúng/Sai":
            generation = self.rag_chain_tf.invoke({"context": documents, "question": question})
        elif question_type == "Trắc nghiệm":
            generation = self.rag_chain_mc.invoke({"context": documents, "question": question})
        else:
            generation = self.rag_chain_ft.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def retrieve(self, state):
        """
                Retrieve relevant documents for a given question

                Args:
                    state["question"]: The input question
                    k (int): Number of documents to retrieve

                Returns:
                    Dict containing retrieved documents and original question
                """
        print("---RETRIEVE---")
        question = state["question"]
        documents = self.retrieve_tool.similarity_search(query=question, k=3)
        return {"documents": documents, "question": question}

    def grade_documents(self, state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def grade_documents(self,state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def transform_query(self, state):
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        better_question = self.web_question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": question, "query_trans": better_question}

    def web_search(self, state):
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        query_trans = state["query_trans"]
        # Web search
        docs = self.web_search_tool.invoke({"query": query_trans})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"documents": documents, "question": question}

    def decide_to_generate(self, state):
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]

        if web_search == "Yes":
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"