from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from .prompt import REACT_PROMPT


class ReactAgent:
    def __init__(self, llm, tools):
        self.prompt = PromptTemplate.from_template(REACT_PROMPT)
        self.llm = llm
        self.tools = tools

    def react_agent(self):
        search_agent = create_react_agent(self.llm, self.tools, self.prompt)
        agent_executor = AgentExecutor(
            agent=search_agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5,
            max_execution_time=None,
            early_stopping_method="force",
            handle_parsing_errors=self._handle_error,
            trim_intermediate_steps=-1
        )
        return agent_executor

    def _handle_error(self, error) -> str:
        return str(error)[:50]
