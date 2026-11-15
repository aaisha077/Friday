import os
import openai
from dotenv import load_dotenv
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.agent_pipeline import (
    AgentPipeline, SystemMessage, InitQuery, OpenAILLM,
    ToolsExecutor, ToolsExecutionLoop,
)

load_dotenv()
client = openai.OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    timeout=60,
    max_retries=1,
)
llm = OpenAILLM(client, "openai/gpt-oss-20b", reasoning_effort="medium")
pipeline = AgentPipeline([
    SystemMessage("You are a helpful banking assistant. Use the tools to complete the user's request."),
    InitQuery(),
    llm,
    ToolsExecutionLoop([ToolsExecutor(), llm]),
])
pipeline.name = "groq-gpt-oss-20b"
suite = get_suites("v1.2.1")["banking"]