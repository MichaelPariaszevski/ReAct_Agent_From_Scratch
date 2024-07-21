from dotenv import load_dotenv, find_dotenv

from typing import Union
from typing import List

from langchain.agents import tool
from langchain.prompts.prompt import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.agents.format_scratchpad import format_log_to_str

from langchain_openai import ChatOpenAI

from callbacks import AgentCallbackHandler

load_dotenv(find_dotenv(), override=True)


@tool  # This decorator creates a LangChain tool from this function (get_text_length)
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""

    print(f"get_text_length enter with text = {text}")

    text = text.strip("'\n").strip('"')  # This strips away non-alphabetic characters

    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain")
    # print(get_text_length(text="Dog"))
    tools_2 = [get_text_length]

    template = """ 
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    # Thought: {agent_scratchpad} was removed, for now

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools_2),
        tool_names=", ".join([i.name for i in tools_2]),
    )  # .partial() is used because we already know the values that will occupy the variables (tools with tools and tool_names with tools), whereas with PromptTemplate.from_template(template=template, input_variables=["EXAMPLE"]), this would be used when the variables would be defined/occupied by user inputs (input as input)

    # kwargs stands for "keyword arguments"

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        stop=["\nObservation", "Observation"],
        callbacks=[AgentCallbackHandler()]
    )  # stop = ["\nObservation", "Observation"] tells the llm to stop working or stop outputting text once the "\nObservation" or "Observation" token is reached

    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )  # Using LCEL (LangChain Expression Language)

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the text length of 'DOG' in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )

    print(agent_step)
    print("-" * 50)
    print(
        type(agent_step)
    )  # the type is langchain_core.agents.AgentAction (AgentAction object)
    print("-" * 50)
    print(agent_step.tool)
    print("-" * 50)
    print(agent_step.tool_input)
    print("-" * 50)
    print(agent_step.log)
    # print("-" * 50)
    # print(agent_step.AgentFinish)
    print("-" * 100)
    # print(res.content)

    if isinstance(agent_step, AgentAction):
        tool_name_2 = agent_step.tool
        tool_to_use = find_tool_by_name(tools=tools_2, tool_name=tool_name_2)
        input_value = agent_step.tool_input

        observation = tool_to_use.func(str(input_value))

        print("-" * 150)
        print(f"observation = {observation}")
        print("-" * 150)

        intermediate_steps.append((agent_step, str(observation)))

    agent_step_2: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the text length of 'DOG' in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )

    print("-" * 100)
    print(agent_step_2)
    print("-" * 50)
    print(
        type(agent_step_2)
    )  # here, the type is langchain_core.agents.AgentFinish (AgentFinish object)
    print("-" * 50)
    print(agent_step_2.return_values)
    print("-" * 50)
    print(agent_step_2.return_values["output"])
    print("-" * 50)
    print(agent_step_2.log)
    print("-" * 50)
    print("-" * 150)
    # print(agent_step_2)

    if isinstance(agent_step_2, AgentFinish):
        print(agent_step_2.return_values)
