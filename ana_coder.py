from openai import OpenAI
class LLM:
    def __init__(self,model: str = "gpt-3.5-turbo"):
        """
        Initializes the LLM with the OpenAI API key and model.
        :param api_key: Your OpenAI API key.
        :param model: The OpenAI model to use for language processing.
        """
        self.client = OpenAI(api_key="put the api key here")
        self.model = model
    def generate_text(self, prompt: str) -> str:
        """
        Generates text based on the provided prompt using the OpenAI API.
        :param prompt: The prompt for text generation.
        :return: A string of the generated text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a proffesional coding assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content
            return text.strip()
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""

class CodeCreatorAgent:
    def __init__(self, language: str, llm: LLM):
        """
        Initializes the CodeCreatorAgent with language and LLM instance.
        :param language: The programming language for code generation.
        :param llm: An instance of the LLM class for interacting with the OpenAI API.
        """
        self.language = language
        self.llm = llm

    def generate_code(self, requirements: dict) -> str:
        """
        Generates code based on the provided requirements using the LLM.
        :param requirements: A dictionary containing the requirements for the code.
        :return: A string of the generated code.
        """
        # Convert the requirements dictionary to a string for the prompt
        requirements_str = "\n".join([f"{key}: {value}" for key, value in requirements.items()])
        print(requirements_str)
        
        # Define the prompt for the code generation
        prompt = f"Generate {self.language} code based on the following requirements:\n{requirements_str}"
        print(prompt)
        
        # Use the LLM instance to generate the code
        code = self.llm.generate_text(prompt)
        return code

class CodeTestGeneratorAgent:
    def __init__(self, language: str):
        self.language = language

    def generate_tests(self, code: str) -> str:
        """
        Generates test cases for the provided code.
        :param code: The code snippet to generate tests for.
        :return: A string of generated test cases.
        """
        # Test generation logic here
        tests = f"Generated tests for {self.language} code: {code}"
        return tests
    

class CodeExecutorAgent:
    def __init__(self):
        pass

    def execute_code(self, code: str) -> str:
        """
        Executes the provided code.
        :param code: The code snippet to execute.
        :return: The result of the execution.
        """
        # Code execution logic here
        result = f"Execution result for code: {code}"
        return result

    def execute_tests(self, tests: str) -> str:
        """
        Executes the provided tests.
        :param tests: The test cases to execute.
        :return: The result of the test execution.
        """
        # Test execution logic here
        result = f"Test results: {tests}"
        return result
    

"""
what the Insulock code is doing :
import code pandas ,  numpy , sklearn , torch , tqdm , copy , matplotlib , numpy , 


"""


if __name__ == "__main__":
    # api_key = "sk-None-ncOVWoFduEAqZP0u0OAJT3BlbkFJofD9Vs1UlYhd5i6ezPFZ"  # Replace with your OpenAI API key
    llm = LLM()
    
    code_creator = CodeCreatorAgent(language='Python', llm=llm)
    
    requirements = {
        "Function": "Calculate factorial",
        "Language": "Python",
        "Additional Information": "The function should handle non-negative integers."
    }
    
    generated_code = code_creator.generate_code(requirements)
    print("Generated Code:")
    print(generated_code)