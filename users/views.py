from django.shortcuts import render
from django.http import JsonResponse
# from AiModels.FLOW.tools import initialize_qdrant_hybrid, initialize_tavily_tool
# from langchain_groq import ChatGroq
# from AiModels.FLOW.react import  ReactAgent
# from AiModels.FLOW.config import GROQ_API_KEY, LANGSMITH_KEY, GENERATIVE_MODEL_NAME
from AI_Model.tools import initialize_qdrant_hybrid, initialize_tavily_tool
from langchain_groq import ChatGroq
from AI_Model.react import  ReactAgent
from AI_Model.config import GROQ_API_KEY, LANGSMITH_KEY, GENERATIVE_MODEL_NAME
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
import json
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_tool_calling_agent

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Vietnamese assistant.Please answer in Vietnamese.\n\nYou may not need to use tools for every query - the user may just want to chat!"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
system = """Determine whether the user's question is a general question or a law-related question..
Give a binary value 'yes' or 'no' to indicate whether it is a general question."""

class ClsQuestion(BaseModel):
    """Binary score for determining if a question is general or law-related."""

    binary_score: str = Field(
        description="Indicates whether the question is general. Returns 'yes' for general questions and 'no' for law-related questions."
    )
llm_deepseek = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)
structured_llm_cls = llm_deepseek.with_structured_output(ClsQuestion)

cls_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),
    ]
)
cls = cls_prompt | structured_llm_cls


qdrant_hybrid = initialize_qdrant_hybrid()
tavily_tool = initialize_tavily_tool()


llm = ChatGroq(model=GENERATIVE_MODEL_NAME, api_key=GROQ_API_KEY)
agent_1 = create_tool_calling_agent(llm, [tavily_tool], prompt)
def _handle_error(self, error) -> str:
    return str(error)[:50]
agent_normal = AgentExecutor(agent=agent_1, tools=[tavily_tool], verbose=True,return_intermediate_steps=True,
            max_iterations=5,
            max_execution_time=None,
            early_stopping_method="force",
            handle_parsing_errors=_handle_error,
            trim_intermediate_steps=-1)


def get_relevant_document(query: str) -> str:
    total_content = ""
    results = qdrant_hybrid.similarity_search(query=query, k=3)
    for doc in results:
        total_content += doc.page_content + "\n"
    return total_content


# Create structured tool
get_relevant_document_tool = StructuredTool.from_function(
    name="Get Relevant document",
    func=get_relevant_document,
    description="Useful for getting relevant documents from local Qdrant store"
)

# Initialize AI model

workflow = ReactAgent(llm, [get_relevant_document_tool, tavily_tool]).react_agent()
# workflow = (llm, qdrant_hybrid, tavily_tool).rag_flow()
# query = "Người nghiện ma túy từ đủ 18 tuổi trở lên bị áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc theo quy định của Luật Xử lý vi phạm hành chính khi bị phát hiện sử dụng chất ma túy một cách trái phép trong thời gian cai nghiện ma túy tự nguyện, đúng hay sai?"
# response = workflow.invoke({"question": query, "question_type": "Tự luận"})
# print(response)
def chat(request):
    return render(request, 'chat.html')

def account(request):
    return render(request, 'account.html')
def get_ai_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')

            # Classify if the question is general or law-related
            ans = cls.invoke({"question": user_message}).binary_score

            # Initialize variables that will be used for response structure
            combined_response = ""
            references = []

            if ans == "yes":
                # For general questions, use agent_normal
                response = agent_normal.invoke({"input": user_message})

                # Extract the final answer
                final_answer = response['output']
                combined_response = final_answer

                # Extract any references from tavily tool if used
                if "intermediate_steps" in response:
                    for step in response["intermediate_steps"]:
                        if isinstance(step, tuple) and len(step) == 2:
                            action = step[0]
                            result = step[1]
                            if action.tool == 'tavily_search_results_json':
                                for search_result in result:
                                    references.append({
                                        'type': 'link',
                                        'url': search_result["url"],
                                        'text': search_result["content"]
                                    })
            else:
                # For law-related questions, use workflow with ReAct agent
                question_type = "Tự luận"  # Default question type
                response = workflow.invoke({
                    "question": user_message,
                    "question_type": question_type
                })

                # Extract the final answer
                final_answer = response['output']

                # Only add reasoning process if intermediate_steps exist and are not empty
                if "intermediate_steps" in response and response["intermediate_steps"]:
                    # Extract the ReAct reasoning steps
                    react_steps = []
                    for step in response["intermediate_steps"]:
                        if isinstance(step, tuple) and len(step) == 2:
                            # Format action and result
                            action = step[0]
                            result = step[1]
                            react_steps.append({
                                "thought": action.log,  # This should contain the Thought, Action, Action Input
                                "result": result
                            })

                    # Only display reasoning process if we have valid steps
                    if react_steps:
                        # Format the react steps as markdown
                        react_markdown = ""
                        for i, step in enumerate(react_steps):
                            react_markdown += f"### Bước {i + 1}\n"
                            react_markdown += f"**Quá Trình Nghĩ:**\n```\n{step['thought']}\n```\n\n"
                            react_markdown += f"**Quan Sát:**\n```\n{step['result']}\n```\n\n"

                        # Combine the final answer with the reasoning steps
                        combined_response = f"{final_answer}\n\n---\n\n### Quá Trình Suy Luận:\n{react_markdown}"
                    else:
                        combined_response = final_answer
                else:
                    combined_response = final_answer

                # Prepare references
                if "intermediate_steps" in response:
                    for step in range(len(response["intermediate_steps"])):
                        if response["intermediate_steps"][step][0].tool == 'tavily_search_results_json':
                            for result in response["intermediate_steps"][step][1]:
                                references.append({'type': 'link', 'url': result["url"], "text": result["content"]})
                        elif response["intermediate_steps"][step][0].tool == "Get Relevant document":
                            references.append({'type': 'modal', 'content': response["intermediate_steps"][step][1]})

            return JsonResponse({
                'status': 'success',
                'message': combined_response,
                'references': references,
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e),
                'references': []
            }, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)
