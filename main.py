from agents import Agent ,Runner , AsyncOpenAI ,OpenAIChatCompletionsModel ,RunConfig ,function_tool ,enable_verbose_stdout_logging
from dotenv import load_dotenv
import os

load_dotenv()
enable_verbose_stdout_logging()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

external_client = AsyncOpenAI(
    api_key= GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client,
 )

config = RunConfig(
     model=model,
     model_provider=external_client,
     tracing_disabled=True,
)

@function_tool
def fetch_weather(city: str) :
    """
    Fetches the current weather for a given city.

    Args :
    city :(str)
    """

    return f"The weather in {city} is sunny"

weather_assistant = Agent(
    name="Weather Assistant",
    instructions="You are a helpful assistant that provides weather information.",
    tools=[fetch_weather],
)

print("tools>>>",weather_assistant.tools[0])


result = Runner.run_sync(
    weather_assistant,
    input="What is the weather in Karachi?",
    run_config=config,)
    

print("result>>>", result.final_output)

if __name__ == "__main__":
    print("Run completed successfully.")
    print("Final output:", result.final_output)
  
