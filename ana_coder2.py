from openai import OpenAI
import concurrent.futures
import re
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

    def generate_code(self, requirements_str: str) -> str:
        """
        use scikit-learn
        Generates code based on the provided requirements using the LLM.
        :param requirements: A dictionary containing the requirements for the code.
        :return: A string of the generated code.
        """
        # Define the prompt for the code generation
        prompt = f"""
        {few_shot_examples}\n\n\nGenerate {self.language} program that meets the specified requirements:\n{requirements_str} .

        Ensure to include any necessary installs in the code for external libraries used.
        please install scikit-learn instead of sklearn also for all the installation libraries please install the latest version and correct library names
        Add the code provided below in the program and call the libraries to be installed inside the install function.
        ```python
        import subprocess
        import sys

        if the library is sklearn use scikit-learn instead

        def install(package):
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        # Example usage
        install('numpy')
        install('pandas')
        install('scikit-learn')
        ```
        
        """
        
        # Use the LLM instance to generate the code
        code = self.llm.generate_text(prompt)
        return extract_python_code(code)
    
    def regenerate_code(self, requirements_str: str, code_result: str, test_result: str) -> str:
        """
        Generates test cases for the provided code.
        :param code: The code snippet to generate tests for.
        :return: A string of generated test cases.
        """
        # Define the prompt for the tests' generation
        prompt = f"""
        {few_shot_examples}\n\n\n
        Based on the results provided below, please revise and improve the program to ensure it functions correctly. 
        Generate a new {self.language} program that meets the specified requirements:\n{requirements_str}.
        Do not generate the tests itself in the output, only the {self.language} program.

        Ensure to include any necessary installs in the code for external libraries used.
        Add the code provided below in the program and call the libraries to be installed inside the install function.
        ```python
        import subprocess
        import sys

        def install(package):
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        # Example usage
        install('numpy')
        install('pandas')
        install('scikit-learn')
        ```

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

    def generate_tests(self, requirements_str: str) -> str:
        """
        Generates test cases for the provided code.
        :param code: The code snippet to generate tests for.
        :return: A string of generated test cases.
        """
        # Define the prompt for the tests' generation
        prompt = f"Generate {self.language} tests for a {self.language} program that implements the following requirements:\n{requirements_str}. Do not generate the program itself in the output, only the {self.language} tests. Ensure to include any necessary %pip installs in the code for external libraries used."
        # Use the LLM instance to generate the tests
        tests = self.llm.generate_text(prompt)
        return extract_python_code(tests)
    
    def regenerate_tests(self, requirements_str: str, code_result: str, test_result: str) -> str:
        """
        Generates test cases for the provided code.
        :param code: The code snippet to generate tests for.
        :return: A string of generated test cases.
        """
        # Define the prompt for the tests' generation
        prompt = f"""
        Based on the results below, please revise and improve the test cases for the program.
        Generate again only the {self.language} tests for a {self.language} program that implements the following requirements:\n{requirements_str}.
        Do not generate the program itself in the output, only the {self.language} tests.

        Ensure to include any necessary installs in the code for external libraries used.
        Add the code provided below in the program and call the libraries to be installed inside the install function.
        ```python
        import subprocess
        import sys

        def install(package):
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        # Example usage
        install('numpy')
        install('pandas')
        install('scikit-learn')
        ```

        code_result: {code_result}
        test_result: {test_result}
        """
        # Use the LLM instance to generate the tests
        tests = self.llm.generate_text(prompt)
        return extract_python_code(tests)
    

class CodeExecutorAgent:
    def __init__(self):
        pass

    def execute_code(self, code: str) -> str:
        """
        Executes the provided code.
        :param code: The code snippet to execute.
        :return: The result of the execution.
        """
        # Assuming we have a method to execute code based on the language
        try:
            exec_globals = {}
            exec(code, exec_globals)
            result = "Passed"
        except Exception as e:
            result = f"Error during code execution: {e}"
        return result

    def execute_tests(self, code: str, tests: str) -> str:
        """
        Executes the provided tests.
        :param code: The code snippet to test.
        :param tests: The test cases to execute.
        :return: The result of the test execution.
        """
        # Combine the function code and the test code
        combined_code = code + "\n" + tests
        exec_globals = {}
        try:
            exec(combined_code, exec_globals)
            result = "Passed"
        except AssertionError as e:
            result = f"Test failed: {e}"
        except Exception as e:
            result = f"Error during test execution: {e}"
        return result


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

























if __name__ == "__main__":
    # requirements = {
    #     "Function": "Calculate factorial",
    #     "Function Name": "factorial",
    #     "Language": "Python",
    #     "Additional Information": "The function should handle non-negative integers."
    # }
    api_key_creator = "place"
    llm_planner = LLM(api_key_creator)
    planner_agent = PlannerAgent(llm_planner)

    query = "Generate detailed requirements for analaysis (split , proprocess) for genomic data and then "
    requirements = planner_agent.plan(query)
    print(requirements)
    # requirements_str = "\n".join([f"{key}: {value}" for key, value in requirements.items()])

    requirements_str = requirements


    llm_creator = LLM(api_key_creator)
    code_creator = CodeCreatorAgent(language='Python', llm=llm_creator)

    api_key_tester = ""
    llm_tester = LLM(api_key_tester)
    code_tester = CodeTestGeneratorAgent(language='Python', llm=llm_tester)

    code_executor = CodeExecutorAgent()
    
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_code = executor.submit(code_creator.generate_code, requirements_str)
        future_tests = executor.submit(code_tester.generate_tests, requirements_str)
        
        generated_code = future_code.result()
        generated_tests = future_tests.result()

    print("Generated Code:")
    print(generated_code)

    print("Generated Tests:")
    print(generated_tests)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_code_result = executor.submit(code_executor.execute_code, generated_code)
        future_tests_result = executor.submit(code_executor.execute_tests, generated_code, generated_tests)
        
        code_result = future_code_result.result()
        tests_result = future_tests_result.result()

    print("Code Result:")
    print(code_result)

    print("Tests Result:")
    print(tests_result)

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
    print(generated_code)

    print("Code Result:")
    print(code_result)

    print("Generated Tests:")
    print(generated_tests)

    print("Tests Result:")
    print(tests_result)

with open('final_output.txt', 'w') as file:
    file.write("Generated Code:\n")
    file.write(generated_code)
    file.write("\n\nGenerated Tests:\n")
    file.write(generated_tests)
    file.write("\n\nCode Result:\n")
    file.write(code_result)
    file.write("\n\nTests Result:\n")
    file.write(tests_result)
























# requirements = {
#     "Class": "Process and split genomic data",
#     "Class Name": "GenomicDataProcessor",
#     "Language": "Python",
#     "Additional Information": "The class should include methods to load, preprocess, and split genomic data suitable for input into a neural network. It should handle various file formats and include options for different preprocessing techniques.",
#     "Class": "Neural Network with LSTM",
#     "Class Name": "NNetwork_wLSTM",
#     "Language": "Python",
#     "Additional Information": "The class should implement a neural network with the following structure: three convolutional layers followed by max pooling layers, a bidirectional LSTM, and several linear layers. The network should process input sequences through these layers, with dropout applied at certain points. The forward method should take two inputs, process them separately through the network, and return the averaged output after applying the final activation function.",
#     "Components": [
#         {
#             "Layer Type": "Convolutional",
#             "Parameters": {
#                 "in_channels": 4,
#                 "out_channels": 160,
#                 "kernel_size": 31
#             }
#         },
#         {
#             "Layer Type": "Max Pooling",
#             "Parameters": {
#                 "kernel_size": 2,
#                 "stride": 2
#             }
#         },
#         {
#             "Layer Type": "Convolutional",
#             "Parameters": {
#                 "in_channels": 160,
#                 "out_channels": 160,
#                 "kernel_size": 20
#             }
#         },
#         {
#             "Layer Type": "Max Pooling",
#             "Parameters": {
#                 "kernel_size": 2,
#                 "stride": 2
#             }
#         },
#         {
#             "Layer Type": "Convolutional",
#             "Parameters": {
#                 "in_channels": 160,
#                 "out_channels": 160,
#                 "kernel_size": 6
#             }
#         },
#         {
#             "Layer Type": "Max Pooling",
#             "Parameters": {
#                 "kernel_size": 8,
#                 "stride": 6
#             }
#         },
#         {
#             "Layer Type": "Bidirectional LSTM",
#             "Parameters": {
#                 "input_size": 160,
#                 "hidden_size": 160,
#                 "num_layers": 2,
#                 "batch_first": True,
#                 "dropout": 0.5,
#                 "bidirectional": True
#             }
#         },
#         {
#             "Layer Type": "Dropout",
#             "Parameters": {
#                 "p": 0.3
#             }
#         },
#         {
#             "Layer Type": "Linear",
#             "Parameters": {
#                 "in_features": 79*320,
#                 "out_features": 925
#             }
#         },
#         {
#             "Layer Type": "Linear",
#             "Parameters": {
#                 "in_features": 925,
#                 "out_features": 925
#             }
#         },
#         {
#             "Layer Type": "Linear",
#             "Parameters": {
#                 "in_features": 925,
#                 "out_features": 1
#             }
#         }
#     ],
#     "Methods": [
#         {
#             "Method Name": "forward_one",
#             "Description": "Processes a single input sequence through the convolutional layers, LSTM, and fully connected layers.",
#             "Inputs": [
#                 {
#                     "Name": "input",
#                     "Type": "Tensor"
#                 }
#             ],
#             "Outputs": [
#                 {
#                     "Name": "x",
#                     "Type": "Tensor"
#                 }
#             ]
#         },
#         {
#             "Method Name": "forward",
#             "Description": "Processes two input sequences through the network and returns the averaged output after applying the final activation function.",
#             "Inputs": [
#                 {
#                     "Name": "x1",
#                     "Type": "Tensor"
#                 },
#                 {
#                     "Name": "x2",
#                     "Type": "Tensor"
#                 }
#             ],
#             "Outputs": [
#                 {
#                     "Name": "out",
#                     "Type": "Tensor"
#                 }
#             ]
#         }
#     ],
#     "Classes": [
#         {
#             "Class Name": "NNetwork",
#             "Description": "A neural network class with convolutional layers, designed for processing genomic sequences without LSTM layers. It includes methods for forward pass using two separate inputs.",
#             "Components": [
#                 {
#                     "Layer Type": "Convolutional",
#                     "Parameters": {
#                         "in_channels": 4,
#                         "out_channels": 160,
#                         "kernel_size": 31
#                     }
#                 },
#                 {
#                     "Layer Type": "Max Pooling",
#                     "Parameters": {
#                         "kernel_size": 2,
#                         "stride": 2
#                     }
#                 },
#                 {
#                     "Layer Type": "Convolutional",
#                     "Parameters": {
#                         "in_channels": 160,
#                         "out_channels": 160,
#                         "kernel_size": 20
#                     }
#                 },
#                 {
#                     "Layer Type": "Max Pooling",
#                     "Parameters": {
#                         "kernel_size": 2,
#                         "stride": 2
#                     }
#                 },
#                 {
#                     "Layer Type": "Convolutional",
#                     "Parameters": {
#                         "in_channels": 160,
#                         "out_channels": 160,
#                         "kernel_size": 6
#                     }
#                 },
#                 {
#                     "Layer Type": "Max Pooling",
#                     "Parameters": {
#                         "kernel_size": 8,
#                         "stride": 6
#                     }
#                 },
#                 {
#                     "Layer Type": "Dropout",
#                     "Parameters": {
#                         "p": 0.3
#                     }
#                 },
#                 {
#                     "Layer Type": "Linear",
#                     "Parameters": {
#                         "in_features": 79*160,
#                         "out_features": 925
#                     }
#                 },
#                 {
#                     "Layer Type": "Linear",
#                     "Parameters": {
#                         "in_features": 925,
#                         "out_features": 925
#                     }
#                 },
#                 {
#                     "Layer Type": "Linear",
#                     "Parameters": {
#                         "in_features": 925,
#                         "out_features": 1
#                     }
#                 }
#             ],
#             "Methods": [
#                 {
#                     "Method Name": "forward_one",
#                     "Description": "Processes a single input sequence through the convolutional layers and fully connected layers.",
#                     "Inputs": [
#                         {
#                             "Name": "input",
#                             "Type": "Tensor"
#                         }
#                     ],
#                     "Outputs": [
#                         {
#                             "Name": "x",
#                             "Type": "Tensor"
#                         }
#                     ]
#                 },
#                 {
#                     "Method Name": "forward",
#                     "Description": "Processes two input sequences through the network and returns the averaged output after applying the final activation function.",
#                     "Inputs": [
#                         {
#                             "Name": "x1",
#                             "Type": "Tensor"
#                         },
#                         {
#                             "Name": "x2",
#                             "Type": "Tensor"
#                         }
#                     ],
#                     "Outputs": [
#                         {
#                             "Name": "out",
#                             "Type": "Tensor"
#                         }
#                     ]
#                 }
#             ]
#         }
#     ],
#     "Function": [
#         {
#             "Function Name": "count_parameters",
#             "Language": "Python",
#             "Description": "Counts the number of trainable parameters in the given model.",
#             "Parameters": [
#                 {
#                     "Name": "model",
#                     "Type": "nn.Module",
#                     "Description": "The PyTorch model for which the parameters are to be counted."
#                 }
#             ],
#             "Returns": [
#                 {
#                     "Name": "parameter_count",
#                     "Type": "int",
#                     "Description": "The total number of trainable parameters in the model."
#                 }
#             ],
#             "Usage": [
#                 {
#                     "Code": "print('Number of Parameters - NNetwork_wLSTM : ', count_parameters(NNetwork_wLSTM()))",
#                     "Description": "Prints the number of trainable parameters in the NNetwork_wLSTM model."
#                 },
#                 {
#                     "Code": "print('Number of Parameters - NNetwork : ', count_parameters(NNetwork()))",
#                     "Description": "Prints the number of trainable parameters in the NNetwork model."
#                 }
#             ]
#         }
#     ],
#     "Function": [
#         {
#             "Function Name": "prcs",
#             "Language": "Python",
#             "Description": "Computes the area under the precision-recall curve.",
#             "Parameters": [
#                 {
#                     "Name": "y",
#                     "Type": "list",
#                     "Description": "List or array of true binary labels."
#                 },
#                 {
#                     "Name": "y_proba",
#                     "Type": "list",
#                     "Description": "List or array of predicted probabilities."
#                 }
#             ],
#             "Returns": [
#                 {
#                     "Name": "lr_auc",
#                     "Type": "float",
#                     "Description": "The area under the precision-recall curve."
#                 }
#             ],
#             "Usage": [
#                 {
#                     "Code": "prc = prcs(y_true, y_proba)",
#                     "Description": "Calculates the precision-recall curve score using true labels and predicted probabilities."
#                 }
#             ]
#         },
#         {
#             "Function Name": "rocs",
#             "Language": "Python",
#             "Description": "Computes the ROC AUC score.",
#             "Parameters": [
#                 {
#                     "Name": "y",
#                     "Type": "list",
#                     "Description": "List or array of true binary labels."
#                 },
#                 {
#                     "Name": "y_proba",
#                     "Type": "list",
#                     "Description": "List or array of predicted probabilities."
#                 }
#             ],
#             "Returns": [
#                 {
#                     "Name": "lr_auc",
#                     "Type": "float",
#                     "Description": "The ROC AUC score."
#                 }
#             ],
#             "Usage": [
#                 {
#                     "Code": "roc = rocs(y_true, y_proba)",
#                     "Description": "Calculates the ROC AUC score using true labels and predicted probabilities."
#                 }
#             ]
#         }
#     ],
#     "Class": [
#         {
#             "Class Name": "Trainer",
#             "Language": "Python",
#             "Additional Information": "The Trainer class handles the training and evaluation of a PyTorch model, including data loading, optimization, and performance metrics computation.",
#             "Components": [
#                 {
#                     "Component": "DataLoader",
#                     "Parameters": {
#                         "train_data": "Training dataset",
#                         "val_data": "Validation dataset",
#                         "batch_size": "Batch size for data loading"
#                     }
#                 },
#                 {
#                     "Component": "Model",
#                     "Parameters": {
#                         "model": "PyTorch model to be trained"
#                     }
#                 },
#                 {
#                     "Component": "Optimizer",
#                     "Parameters": {
#                         "learning_rate": "Learning rate for optimization",
#                         "weight_decay": "Weight decay for regularization"
#                     }
#                 },
#                 {
#                     "Component": "Loss Function",
#                     "Parameters": {
#                         "criterion": "Loss function used for training (BCELoss)"
#                     }
#                 }
#             ],
#             "Methods": [
#                 {
#                     "Method Name": "train",
#                     "Description": "Trains the model for a specified number of epochs and evaluates its performance.",
#                     "Inputs": [],
#                     "Outputs": [],
#                     "Usage": [
#                         {
#                             "Code": "trainer.train()",
#                             "Description": "Trains the model using the provided training and validation data."
#                         }
#                     ]
#                 }
#             ]
#         }
#     ],
#     "Function": [
#         {
#             "Function Name": "prcs",
#             "Language": "Python",
#             "Description": "Computes the area under the precision-recall curve.",
#             "Parameters": [
#                 {
#                     "Name": "y",
#                     "Type": "list",
#                     "Description": "List or array of true binary labels."
#                 },
#                 {
#                     "Name": "y_proba",
#                     "Type": "list",
#                     "Description": "List or array of predicted probabilities."
#                 }
#             ],
#             "Returns": [
#                 {
#                     "Name": "lr_auc",
#                     "Type": "float",
#                     "Description": "The area under the precision-recall curve."
#                 }
#             ],
#             "Usage": [
#                 {
#                     "Code": "prc = prcs(y_true, y_proba)",
#                     "Description": "Calculates the precision-recall curve score using true labels and predicted probabilities."
#                 }
#             ]
#         },
#         {
#             "Function Name": "rocs",
#             "Language": "Python",
#             "Description": "Computes the ROC AUC score.",
#             "Parameters": [
#                 {
#                     "Name": "y",
#                     "Type": "list",
#                     "Description": "List or array of true binary labels."
#                 },
#                 {
#                     "Name": "y_proba",
#                     "Type": "list",
#                     "Description": "List or array of predicted probabilities."
#                 }
#             ],
#             "Returns": [
#                 {
#                     "Name": "lr_auc",
#                     "Type": "float",
#                     "Description": "The ROC AUC score."
#                 }
#             ],
#             "Usage": [
#                 {
#                     "Code": "roc = rocs(y_true, y_proba)",
#                     "Description": "Calculates the ROC AUC score using true labels and predicted probabilities."
#                 }
#             ]
#         }
#     ],
#     "Class": [
#         {
#             "Class Name": "Trainer",
#             "Language": "Python",
#             "Additional Information": "The Trainer class handles the training and evaluation of a PyTorch model, including data loading, optimization, and performance metrics computation.",
#             "Components": [
#                 {
#                     "Component": "DataLoader",
#                     "Parameters": {
#                         "train_data": "Training dataset",
#                         "val_data": "Validation dataset",
#                         "batch_size": "Batch size for data loading"
#                     }
#                 },
#                 {
#                     "Component": "Model",
#                     "Parameters": {
#                         "model": "PyTorch model to be trained"
#                     }
#                 },
#                 {
#                     "Component": "Optimizer",
#                     "Parameters": {
#                         "learning_rate": "Learning rate for optimization",
#                         "weight_decay": "Weight decay for regularization"
#                     }
#                 },
#                 {
#                     "Component": "Loss Function",
#                     "Parameters": {
#                         "criterion": "Loss function used for training (BCELoss)"
#                     }
#                 }
#             ],
#             "Methods": [
#                 {
#                     "Method Name": "train",
#                     "Description": "Trains the model for a specified number of epochs and evaluates its performance.",
#                     "Inputs": [],
#                     "Outputs": [],
#                     "Usage": [
#                         {
#                             "Code": "trainer.train()",
#                             "Description": "Trains the model using the provided training and validation data."
#                         }
#                     ]
#                 }
#             ]
#         }
        
#     ],
#     "Tester": {
#         "Component Name": "Tester",
#         "Language": "Python",
#         "Description": "Tests a PyTorch model on given test data and returns predictions and true labels.",
#         "Parameters": [
#             {
#                 "Name": "model",
#                 "Type": "nn.Module",
#                 "Description": "PyTorch model to be tested."
#             },
#             {
#                 "Name": "model_weight_path",
#                 "Type": "string",
#                 "Description": "Path to the saved model weights."
#             },
#             {
#                 "Name": "test_data",
#                 "Type": "Dataset",
#                 "Description": "Test data to be evaluated."
#             },
#             {
#                 "Name": "batch_size",
#                 "Type": "int",
#                 "Description": "Batch size for testing."
#             }
#         ],
#         "Methods": [
#             {
#                 "Method Name": "test",
#                 "Description": "Tests the model and returns true labels and predicted values.",
#                 "Inputs": [],
#                 "Outputs": [
#                     {
#                         "Name": "y_true",
#                         "Type": "numpy array",
#                         "Description": "Array of true labels."
#                     },
#                     {
#                         "Name": "y_pred",
#                         "Type": "numpy array",
#                         "Description": "Array of predicted values."
#                     }
#                 ]
#             }
#         ]
#     },
#     "plot_prc_roc": {
#         "Component Name": "plot_prc_roc",
#         "Language": "Python",
#         "Description": "Plots Precision-Recall and ROC curves for model evaluation.",
#         "Parameters": [
#             {
#                 "Name": "y_true",
#                 "Type": "numpy array",
#                 "Description": "Array of true labels for the overall dataset."
#             },
#             {
#                 "Name": "y_pred",
#                 "Type": "numpy array",
#                 "Description": "Array of predicted values for the overall dataset."
#             },
#             {
#                 "Name": "y_true1",
#                 "Type": "numpy array",
#                 "Description": "Array of true labels for type 1 dataset."
#             },
#             {
#                 "Name": "y_pred1",
#                 "Type": "numpy array",
#                 "Description": "Array of predicted values for type 1 dataset."
#             },
#             {
#                 "Name": "y_true2",
#                 "Type": "numpy array",
#                 "Description": "Array of true labels for type 2 dataset."
#             },
#             {
#                 "Name": "y_pred2",
#                 "Type": "numpy array",
#                 "Description": "Array of predicted values for type 2 dataset."
#             },
#             {
#                 "Name": "y_true3",
#                 "Type": "numpy array",
#                 "Description": "Array of true labels for type 3 dataset."
#             },
#             {
#                 "Name": "y_pred3",
#                 "Type": "numpy array",
#                 "Description": "Array of predicted values for type 3 dataset."
#             },
#             {
#                 "Name": "out_path",
#                 "Type": "string",
#                 "Description": "Path where the plot images should be saved."
#             }
#         ],
#         "Methods": [
#             {
#                 "Method Name": "plot_prc",
#                 "Description": "Plots the Precision-Recall curve for the overall and each type dataset.",
#                 "Inputs": [],
#                 "Outputs": [],
#                 "Usage": [
#                     {
#                         "Code": "plotter.plot_prc()",
#                         "Description": "Generates and displays the Precision-Recall curve plot."
#                     }
#                 ]
#             },
#             {
#                 "Method Name": "plot_roc",
#                 "Description": "Plots the ROC curve for the overall and each type dataset.",
#                 "Inputs": [],
#                 "Outputs": [],
#                 "Usage": [
#                     {
#                         "Code": "plotter.plot_roc()",
#                         "Description": "Generates and displays the ROC curve plot."
#                     }
#                 ]
#             }
#         ]
#     }

# }