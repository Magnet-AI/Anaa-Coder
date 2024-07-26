from openai import OpenAI
import concurrent.futures
import re

few_shot_examples = """
    Task: Implement a class to process and split genomic data.
    Input: path_to_data
    Code:
    ```python
    import pandas as pd
    import numpy as np

    class DataProcessor:
        def __init__(self, path_to_data):
            self.path = path_to_data

        def concat_data(self):
            cell_types_v = ['GM', 'H1', 'K562', 'MCF7']
            positive, type_1_negative, type_2_negative, type_3_negative = [], [], [], []

            for cell_type in cell_types_v:
                positive.append(pd.read_csv(self.path + cell_type + '_insulator_pos_withCTCF.fa', sep=">chr*", header=None, engine='python').values[1::2][:,0])
                type_1_negative.append(pd.read_csv(self.path + cell_type + '_type1.fa', sep=">chr*", header=None, engine='python').values[1::2][:,0])
                type_2_negative.append(pd.read_csv(self.path + cell_type + '_type2.fa', sep=">chr*", header=None, engine='python').values[1::2][:,0])
                type_3_negative.append(pd.read_csv(self.path + cell_type + '_type3.fa', sep=">chr*", header=None, engine='python').values[1::2][:,0])

            return positive, type_1_negative, type_2_negative, type_3_negative

        def split(self, file, size=0.1):
            len_v = int(len(file) * size)
            np.random.seed(42)
            np.random.shuffle(file)
            train, test = file[len_v:], file[:len_v]
            train, val = train[len_v:], train[:len_v]
            return train, test, val
    ```

    Task: Implement a function to compute the reverse complement of a DNA sequence.
    Input: 'ATGC'
    Output: 'GCAT'
    Code:
    ```python
    def RC(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
                  'a': 't', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
    t = ''
    for base in seq:
        t = complement[base] + t
    return t
    ```

    Task: Implement a PyTorch Dataset class for siamese neural network data.
    Input: data, label
    Code:
    ```python
    import torch
    from torch.utils.data import Dataset

    class Data_siam(Dataset):
        def __init__(self, data, label):
            self.data = data
            self.label = label

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            seq = self.data[index]
            rc_seq = RC(seq)
            ctr = 0
            ar1 = np.zeros((2000, 4))
            for base in seq:
                if base == 'A' or base == 'a':
                    ar1[ctr, 0] = 1
                elif base == 'T' or base == 't':
                    ar1[ctr, 1] = 1
                elif base == 'C' or base == 'c':
                    ar1[ctr, 2] = 1
                elif base == 'G' or base == 'g':
                    ar1[ctr, 3] = 1
                ctr += 1

            ar2 = np.zeros((2000, 4))
            ctr = 0
            for base in rc_seq:
                if base == 'A' or base == 'a':
                    ar2[ctr, 0] = 1
                elif base == 'T' or base == 't':
                    ar2[ctr, 1] = 1
                elif base == 'C' or base == 'c':
                    ar2[ctr, 2] = 1
                elif base == 'G' or base == 'g':
                    ar2[ctr, 3] = 1
                ctr += 1

            ar1 = torch.tensor(ar1).float().permute(1, 0)
            ar2 = torch.tensor(ar2).float().permute(1, 0)
            label = torch.tensor(self.label).float()

            return ar1, ar2, label
    ```
    
    Task: Implement a neural network class with LSTM in PyTorch.
    Input: N/A
    Code:
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class NNetwork_wLSTM(nn.Module):
        def __init__(self):
            super(NNetwork_wLSTM, self).__init__()
            self.Conv1 = nn.Conv1d(in_channels=4, out_channels=160, kernel_size=31)
            self.Maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.Conv2 = nn.Conv1d(in_channels=160, out_channels=160, kernel_size=20)
            self.Maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.Conv3 = nn.Conv1d(in_channels=160, out_channels=160, kernel_size=6)
            self.Maxpool3 = nn.MaxPool1d(kernel_size=8, stride=6)
            self.BiLSTM = nn.LSTM(input_size=160, hidden_size=160, num_layers=2,
                                batch_first=True, dropout=0.5, bidirectional=True)
            self.Drop1 = nn.Dropout(p=0.3)
            self.Linear1 = nn.Linear(79*320, 925)
            self.Linear2 = nn.Linear(925, 925)
            self.Linear3 = nn.Linear(925, 1)

        def forward_one(self, input):
            x = self.Conv1(input)
            x = F.relu(x)
            x = self.Maxpool1(x)
            x = self.Conv2(x)
            x = F.relu(x)
            x = self.Maxpool2(x)
            x = self.Conv3(x)
            x = F.relu(x)
            x = self.Maxpool3(x)
            x_x = torch.transpose(x, 1, 2)
            x, (h_n, h_c) = self.BiLSTM(x_x)
            x = x.contiguous().view(-1, 79*320)
            x = self.Drop1(x)
            x = self.Linear1(x)
            x = F.relu(x)
            x = self.Drop1(x)
            x = self.Linear2(x)
            x = F.relu(x)
            x = self.Linear3(x)
            return x

        def forward(self, x1, x2):
            out1 = self.forward_one(x1)
            out2 = self.forward_one(x2)
            out = (out1 + out2) / 2
            return torch.sigmoid(out)
    ```

    Task: Train a neural network model with the given data.
    Input: train_data, val_data, model, num_epochs, batch_size, learning_rate, weight_decay, pretrain_path, model_path
    Code:
    ```python
    class Trainer:
    def __init__(self, train_data, val_data, model, num_epochs, batch_size, learning_rate, weight_decay, pretrain_path, model_path):
        self.train_data = train_data
        self.val_data = val_data
        self.model_path = model_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pretrain_path = pretrain_path
        self.model = model

    def train(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_loader = torch.utils.data.DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size)
        val_loader = torch.utils.data.DataLoader(self.val_data, shuffle=True, batch_size=self.batch_size)
        model = self.model.to(device)
        min_loss = 100
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        for epoch in range(1, self.num_epochs+1):
            print("Epoch {}".format(epoch))
            model.train()
            train_acc = 0
            train_loss = 0
            for data, label in tqdm(train_loader):
                data, label = data.to(device), label.to(device)
                output = model.forward_one(data).squeeze()
                loss = criterion(torch.sigmoid(output), label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                y_pred = (output > 0.5).float()
                train_acc += torch.sum(y_pred == label)

            loss = train_loss / len(train_loader)
            accuracy = int(train_acc / len(train_loader.dataset) * 100)
            print('\nTrain Data: Average Train Loss: {:.4f}, Train Accuracy: {}/{} ({}%)'.format(loss, train_acc, len(train_loader.dataset), accuracy))

            y_true, y_proba, y_pred = [], [], []
            model.eval()
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for data, target in tqdm(val_loader):
                    data, target = data.to(device), target.to(device)
                    output = model.forward_one(data).squeeze()
                    y_hat = output.cpu().numpy()
                    loss = criterion(torch.sigmoid(output), target)
                    val_loss += loss.item() * data.size(0)
                    y_pred_ = (output > 0.5).float()
                    val_accuracy += sum(y_pred_ == target)
                    for i in range(len(y_pred_)):
                        y_true.append(float(target[i]))
                        y_pred.append(float(y_pred_[i]))
                        y_proba.append(float(y_hat[i]))

                loss = val_loss / len(val_loader.dataset)
                accuracy = val_accuracy / len(val_loader.dataset)
                prc = prcs(y_true, y_proba)
                roc = rocs(y_true, y_proba)
                print('Validation -> AUPRC: {:.4f}, AUROC: {:.4f}, Loss: {:.4f}'.format(prc, roc, loss))
                print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

            if loss < min_loss:
                min_loss = loss
                torch.save(model.state_dict(), self.model_path)

"""

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
        Generates code based on the provided requirements using the LLM.
        :param requirements: A dictionary containing the requirements for the code.
        :return: A string of the generated code.
        """
        # Define the prompt for the code generation
        prompt = f"""
        {few_shot_examples}\n\n\nGenerate {self.language} program that meets the specified requirements:\n{requirements_str}.

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

if __name__ == "__main__":
    # requirements = {
    #     "Function": "Calculate factorial",
    #     "Function Name": "factorial",
    #     "Language": "Python",
    #     "Additional Information": "The function should handle non-negative integers."
    # }

    requirements = {
    "Class": "Process and split genomic data",
    "Class Name": "GenomicDataProcessor",
    "Language": "Python",
    "Additional Information": "The class should include methods to load, preprocess, and split genomic data suitable for input into a neural network. It should handle various file formats and include options for different preprocessing techniques."
    }

    requirements_str = "\n".join([f"{key}: {value}" for key, value in requirements.items()])

    api_key_creator = "place"
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

