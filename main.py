import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. SETUP: Load API Key ---
def setup_environment():
    """
    Loads environment variables from a .env file and checks for the Groq API key.
    """
    load_dotenv()
    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY not found in .env file.")
        exit()


# --- 2. THE PROMPT: The Core of the Translator ---
def create_command_prompt_template():
    """
    Creates a detailed prompt template to instruct the LLM on how to format the output.
    This uses a technique called "few-shot prompting" where we provide examples.
    """
    
    # The prompt is a multi-line string that defines the AI's role, instructions,
    # and provides examples of the desired input/output format.
    prompt_template = """
You are a specialized AI assistant. Your purpose is to convert natural language text into a structured JSON command.
You must analyze the user's request and translate it into a JSON object containing a specific 'action' and its corresponding 'parameters'.

Your response MUST be ONLY the JSON object itself, with no additional text, explanations, or markdown formatting.

---
Here are some examples to guide you:

User Request: Turn on the living room lights and set them to blue.
JSON Command: {{"action": "set_light_state", "location": "living_room", "parameters": {{"state": "on", "color": "blue"}}}}

User Request: Play the 'Chill Hits' playlist on Spotify.
JSON Command: {{"action": "play_music", "service": "spotify", "parameters": {{"playlist": "Chill Hits"}}}}

User Request: What's the weather like in Paris tomorrow?
JSON Command: {{"action": "get_weather", "parameters": {{"location": "Paris, FR", "date": "tomorrow"}}}}

User Request: set a timer for 15 minutes
JSON Command: {{"action": "set_timer", "parameters": {{"duration_minutes": 15}}}}
---

Now, please convert the following user request into a JSON command.

User Request: {user_input}
JSON Command:"""
    
    # We use double curly braces {{ and }} in the examples to "escape" them,
    # so that Python's formatting doesn't confuse them with the main {user_input} variable.
    
    return PromptTemplate(template=prompt_template, input_variables=["user_input"])

# --- 3. THE TRANSLATOR: The Main Application Logic ---
class CommandTranslator:
    def __init__(self, prompt_template):
        """
        Initializes the translator.
        
        Args:
            prompt_template: The LangChain PromptTemplate to use for structuring the LLM's instructions.
        """
        
        # We use a temperature of 0.0 to make the output deterministic.
        # We want precision and reliability, not creativity.
        self.llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.0)
        

        # The '|' (pipe) operator connects the components.
        # It means the output of the prompt flows into the llm,
        # and the output of the llm flows into the string parser.
        self.chain = prompt_template | self.llm | StrOutputParser()

    def translate(self, user_input: str):
        """
        Takes a natural language string and returns a structured JSON command string.
        
        Args:
            user_input: The natural language text from the user.
        
        Returns:
            A string containing the JSON command, or an error message.
        """
        try:
            # We use .invoke() to run the chain. The input is a dictionary
            # where the key matches the input_variable in our prompt.
            json_command = self.chain.invoke({"user_input": user_input})
            return json_command
        except Exception as e:
            # Basic error handling in case the API call fails.
            print(f"An error occurred during translation: {e}")
            return "{\"error\": \"Failed to generate command.\"}"

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # First, make sure our environment and API key are ready.
    setup_environment()
    
    # Create our prompt template and the translator instance.
    command_prompt = create_command_prompt_template()
    translator = CommandTranslator(prompt_template=command_prompt)
    
    print("  Natural Language to Command Interface")
    print("   Enter a command like 'turn off the kitchen lights' or 'exit' to quit.")
    
    # Start the main loop to listen for user input.
    while True:
        # Get input from the user.
        user_query = input("\nYou > ")
        
        # Check if the user wants to exit the program.
        if user_query.lower() == 'exit':
            print(" Goodbye!")
            break
        
        # If not exiting, translate the command and print the result.
        structured_command = translator.translate(user_query)
        
        print(f"  JSON Command:\n{structured_command}")