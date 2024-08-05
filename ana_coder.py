from openai import OpenAI
import concurrent.futures
import re
from e2b_code_interpreter import CodeInterpreter
from prompts import few_shot_examples
import json

class LLM:

    def __init__(self, api_key, model: str = "gpt-4-turbo"):
        """
        Initializes the LLM with the OpenAI API key and model.
        :param api_key: Your OpenAI API key.
        :param model: The OpenAI model to use for language processing.
        """
        self.client = OpenAI(api_key=api_key)
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

    def generate_code(self, requirements: str) -> str:
        """
        Generates code based on the provided requirements using the LLM.
        :param requirements: A dictionary containing the requirements for the code.
        :return: A string of the generated code.
        """
        # Define the prompt for the code generation
        prompt = f"{few_shot_examples}\n\n\nGenerate {self.language} program that meets the specified requirements:\n{requirements}"
        
        # Use the LLM instance to generate the code
        code = self.llm.generate_text(prompt)
        return extract_python_code(code)
    
    def regenerate_code(self, requirements: str, code_result: str, test_result: str) -> str:
        """
        Generates test cases for the provided code.
        :param code: The code snippet to generate tests for.
        :return: A string of generated test cases.
        """
        # Define the prompt for the tests' generation
        prompt = f"""
        {few_shot_examples}\n\n\n
        Based on the results provided below, please revise and improve the program to ensure it functions correctly. 
        Generate a new {self.language} program that meets the specified requirements:\n{requirements}.
        Do not generate the tests itself in the output, only the {self.language} program.

        code_result: {code_result}
        test_result: {test_result}
        """
        # Use the LLM instance to generate the tests
        tests = self.llm.generate_text(prompt)
        return extract_python_code(tests)


class CodeTestGeneratorAgent:

    def __init__(self, language: str, llm: LLM):
        self.language = language
        self.llm = llm

    def generate_tests(self, requirements: str) -> str:
        """
        Generates test cases for the provided code.
        :param code: The code snippet to generate tests for.
        :return: A string of generated test cases.
        """
        # Define the prompt for the tests' generation
        prompt = f"Generate {self.language} tests for a {self.language} program that implements the following requirements:\n{requirements}. Do not generate the program itself in the output, only the {self.language} tests. Ensure to include any necessary %pip installs in the code for external libraries used."
        # Use the LLM instance to generate the tests
        tests = self.llm.generate_text(prompt)
        return extract_python_code(tests)
    
    def regenerate_tests(self, requirements: str, code_result: str, test_result: str) -> str:
        """
        Generates test cases for the provided code.
        :param code: The code snippet to generate tests for.
        :return: A string of generated test cases.
        """
        # Define the prompt for the tests' generation
        prompt = f"""
        Based on the results below, please revise and improve the test cases for the program.
        Generate again only the {self.language} tests for a {self.language} program that implements the following requirements:\n{requirements}.
        Do not generate the program itself in the output, only the {self.language} tests.

        code_result: {code_result}
        test_result: {test_result}
        """
        # Use the LLM instance to generate the tests
        tests = self.llm.generate_text(prompt)
        return extract_python_code(tests)
    

class CodeExecutorAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def execute_code(self, code: str) -> str:
        """
        Executes the provided code using e2b.
        :param code: The code snippet to execute.
        :return: The result of the execution.
        """
        try:
            with CodeInterpreter(api_key=self.api_key) as sandbox:
                execution = sandbox.notebook.exec_cell(code)
                result = "Passed"
        except Exception as e:
            result = f"Error during code execution: {e}"
        return result

    def execute_tests(self, code: str, tests: str) -> str:
        """
        Executes the provided tests using e2b.
        :param code: The code snippet to test.
        :param tests: The test cases to execute.
        :return: The result of the test execution.
        """
        combined_code = code + "\n" + tests
        try:
            with CodeInterpreter(api_key=self.api_key) as sandbox:
                execution = sandbox.notebook.exec_cell(combined_code)
                result = "Passed"
        except AssertionError as e:
            result = f"Test failed: {e}"
        except Exception as e:
            result = f"Error during test execution: {e}"
        return result
    
class PlannerAgent:
    def __init__(self, llm_instance):
        self.llm = llm_instance
    
    def plan(self, query):
        """
        This method takes a query, sends it to the LLM instance, and retrieves a structured plan or requirements.
        """
        prompt = self._generate_prompt(query)
        response = self.llm.generate_text(prompt)
        # print(response)
        return response

    def _generate_prompt(self, query):
        """
        Generates a one-shot prompt for the LLM based on the user's query.
        """
        one_shot_example = (
            "Example Query: 'Generate detailed requirements for a neural network with LSTM for genomic data processing.'\n"
            "Example Response:\n"
            "{\n"
            "  'Class': 'GenomicDataProcessor',\n"
            "  'Class Name': 'GenomicDataProcessor',\n"
            "  'Language': 'Python',\n"
            "  'Additional Information': 'The class should include methods to load, preprocess, and split genomic data suitable for input into a neural network.',\n"
            "  'Components': [\n"
            "    {'Name': 'DataLoader', 'Parameters': ['data_path', 'batch_size']},\n"
            "    {'Name': 'Preprocessor', 'Parameters': ['normalization', 'data_augmentation']}\n"
            "  ],\n"
            "  'Methods': [\n"
            "    {'Name': 'load_data', 'Description': 'Loads data from the specified path.', 'Inputs': ['data_path'], 'Outputs': ['data']},\n"
            "    {'Name': 'preprocess_data', 'Description': 'Preprocesses the data for training.', 'Inputs': ['data'], 'Outputs': ['processed_data']}\n"
            "  ]\n"
            "}\n\n"
            "You are an experienced Bioinformatician who codes neural network models from scratch. You understand the intricacies of genomic data and the specialized requirements for processing and analyzing such data using deep learning techniques. Think carefully about the following aspects:\n"
            "- The structure of the neural network, including layers and connections.\n"
            "- Specific preprocessing steps required for genomic data.\n"
            "- Methods and functions that will be essential for loading, preprocessing, and managing data.\n"
            "- Integration of components to ensure a cohesive workflow for data processing and model training.\n\n"
            f"Generate a detailed plan and requirements for the following query:\n\n{query}\n\n"
            "Include comprehensive information on the class structure, components, methods, and any other relevant details. Ensure that the requirements are precise, actionable, and aligned with best practices for bioinformatics and neural network modeling."
        )
        return one_shot_example

    def _parse_response(self, response):
        """
        Parses the LLM response into a structured format.
        """
        # Assuming the response is in JSON format; adjust as needed
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response. Ensure the response is in valid JSON format.")


def extract_python_code(text):
    """
    Extract the Python code from the provided text string.

    Args:
    text (str): The input text containing the Python code
    
    Returns:
    str: The extracted Python code
    """
    code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    match = code_pattern.search(text)
    
    if match:
        return match.group(1).strip()
    else:
        return "No Python code found in the input text."

if __name__ == "__main__":
    
    # query = "Create a function to calculate the factorial of a number"
    query = "Process and split genomic data. The class should include methods to load, preprocess, and split genomic data suitable for input into a neural network. It should handle various file formats and include options for different preprocessing techniques."

    api_key_planner = "sk-proj-ybl8EvWMQuulGwU8X5RzT3BlbkFJXfFHNPnLUvHJrtbvclzU"
    llm_planner = LLM(api_key_planner)
    code_planner = PlannerAgent(llm_planner)

    api_key_creator = "sk-proj-ybl8EvWMQuulGwU8X5RzT3BlbkFJXfFHNPnLUvHJrtbvclzU"
    llm_creator = LLM(api_key_creator)
    code_creator = CodeCreatorAgent(language='Python', llm=llm_creator)

    api_key_tester = "sk-proj-ybl8EvWMQuulGwU8X5RzT3BlbkFJXfFHNPnLUvHJrtbvclzU"
    llm_tester = LLM(api_key_tester)
    code_tester = CodeTestGeneratorAgent(language='Python', llm=llm_tester)

    code_executor = CodeExecutorAgent("e2b_37bb091311f835a61dd1d96ed32c9d85c9511964")

    requirements = code_planner.plan(query)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_code = executor.submit(code_creator.generate_code, requirements)
        future_tests = executor.submit(code_tester.generate_tests, requirements)
        
        generated_code = future_code.result()
        generated_tests = future_tests.result()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_code_result = executor.submit(code_executor.execute_code, generated_code)
        future_tests_result = executor.submit(code_executor.execute_tests, generated_code, generated_tests)
        
        code_result = future_code_result.result()
        tests_result = future_tests_result.result()

    count = 0

    while ((code_result != "Passed" or tests_result != "Passed") and count < 5):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_code = executor.submit(code_creator.regenerate_code, requirements, code_result, tests_result)   
            future_tests = executor.submit(code_tester.regenerate_tests, requirements, code_result, tests_result)
            
            generated_code = future_code.result()
            generated_tests = future_tests.result()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_code_result = executor.submit(code_executor.execute_code, generated_code)
            future_tests_result = executor.submit(code_executor.execute_tests, generated_code, generated_tests)
            
            code_result = future_code_result.result()
            tests_result = future_tests_result.result()

        count = count + 1

        if (count == 4):
            print("Iteration limit exceeded :(")

    print("Generated Code:")
    print(generated_code  + "\n")

    print("Code Tests:")
    print(generated_tests + "\n")

    print("Code Result:" + code_result  + "\n")

    print("Tests Result:" + tests_result + "\n")

    with open('final_output.txt', 'w') as file:
        file.write("Generated Code:\n")
        file.write(generated_code)
        file.write("\n\nGenerated Tests:\n")
        file.write(generated_tests)
        file.write("\n\nCode Result:\n")
        file.write(code_result)
        file.write("\n\nTests Result:\n")
        file.write(tests_result)