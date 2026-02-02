import os
import langchain
from langchain.agents import initialize_agent
from llama_index.core import ListIndex
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI


class Eunomia:
    def __init__(
        self,
        tools,
        model="gpt-4",
        temp=0.1,
        get_cost=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        **kwargs,
    ):
        self.get_cost = get_cost
        if isinstance(model, str):
            if model.startswith("deepseek"):
                self.llm = ChatOpenAI(
                    model=model,
                    temperature=temp,
                    request_timeout=1000,
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1",
                )
            else:
                self.llm = ChatOpenAI(
                    model=model,
                    temperature=temp,
                    request_timeout=1000,
                )

        else:
            self.llm = model
        # Initialize agent
        memory = ConversationBufferMemory(memory_key="chat_history")
        self.agent_chain = initialize_agent(
            tools,
            self.llm,
            agent=agent_type,
            verbose=True,
            memory=memory,
            stop=None,
            **kwargs,
        )

    def run(self, prompt):
        # ⚠️ DeepSeek does NOT support OpenAI cost callback
        if self.get_cost:
            result = self.agent_chain.run(input=prompt)
            print("⚠️ Cost tracking not supported for DeepSeek")
            return result
        else:
            return self.agent_chain.run(input=prompt)
