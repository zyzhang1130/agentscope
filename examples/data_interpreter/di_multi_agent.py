from typing import Any, Dict, List, Optional, Union, Sequence
import agentscope
from agentscope.agents import ReActAgent
from agentscope.agents.agent import AgentBase
from agentscope.message import Msg
from agentscope.utils import common
from agentscope.models import ModelResponse
from agentscope.pipelines.functional import sequentialpipeline
from agentscope.parsers.json_object_parser import MarkdownJsonObjectParser
from agentscope.service import (
    ServiceToolkit,
    execute_python_code,
    read_json_file,
    write_json_file,
    create_file,
    delete_file,
    move_file,
    create_directory,
    delete_directory,
    move_directory,
    list_directory_content,
    get_current_directory,
    read_text_file,
    write_text_file,
    NoteBookExecutor,
    arxiv_search,
    execute_shell_command,
)
from agentscope.service.service_toolkit import *
from agentscope.service.execute_code.exec_notebook import *


# -*- coding: utf-8 -*-
""" Operators for CSV file and directory. """
import csv
import os
from typing import Any, List

from agentscope.service.service_response import ServiceResponse
from agentscope.service.service_status import ServiceExecStatus


def read_csv_file(file_path: str) -> ServiceResponse:
    """
    Read and parse a CSV file.

    Args:
        file_path (`str`):
            The path to the CSV file to be read.

    Returns:
        `ServiceResponse`: Where the boolean indicates success, the
        Any is the parsed CSV content (typically a list of rows), and the str contains
        an error message if any, including the error type.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            data = [row for row in reader]
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=data,
        )
    except Exception as e:
        error_message = f"{e.__class__.__name__}: {e}"
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content=error_message,
        )


def write_csv_file(
    file_path: str,
    data: List[List[Any]],
    overwrite: bool = False,
) -> ServiceResponse:
    """
    Write data to a CSV file.

    Args:
        file_path (`str`):
            The path to the file where the CSV data will be written.
        data (`List[List[Any]]`):
            The data to write to the CSV file (each inner list represents a row).
        overwrite (`bool`):
            Whether to overwrite the file if it already exists.

    Returns:
        `ServiceResponse`: where the boolean indicates success, and the
        str contains an error message if any, including the error type.
    """
    if not overwrite and os.path.exists(file_path):
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content="FileExistsError: The file already exists.",
        )
    try:
        with open(file_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content="Success",
        )
    except Exception as e:
        error_message = f"{e.__class__.__name__}: {e}"
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content=error_message,
        )
    

import os
import json
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai_api_key  = os.getenv('OPENAI_API_KEY')
dashscope_api_key  = os.getenv('DASHSCOPE_API_KEY')
os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

class PlannerAgent(AgentBase):
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        service_toolkit: ServiceToolkit,
        profile: str = "Planner",
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        self.profile = profile
        self.service_toolkit = service_toolkit
    

    def _extract_task(self, messages: List[Msg]) -> str:
        return messages.content

    def reply(self, messages: List[Msg]) -> Msg:

        
        # task = self._extract_task(messages)
        subtasks = self._decompose_task(messages)
        return subtasks

    def _decompose_task(self, task: str, max_tasks: int = 5) -> List[Dict[str, Any]]:

        # Implement task decomposition
        message = [
            # {
                    # "role": "system",
                    # "content": "You are a helpful assistant.",
                    # },
                    { "role": "user",
                            "content": f"""
                    Task: {task}
                    - Given the task above, break it down into subtasks with dependencies if it is sufficently complex to be solved in one go.
                    - Every subtask should be solvable through either executing code or using tools. The information of all the tools available are here: 
                    {self.service_toolkit.tools_instruction} 
                    - The subtask should not be too simple. If a task can be solve with a single block of code in one go, it should not be broken down further. Example: a subtask cannot be simply installing or importing libraries.
                    - Prioritze using other tools over `execute_python_code` and take the tools available into consideration when decomposing the task. Provide a JSON structure with the following format for the decomposition:
                        ```json
                        [
                            {{
                                "task_id": str = "unique identifier for a task in plan, can be an ordinal",
                                "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
                                "instruction": "what you should do in this task, one short phrase or sentence",
                                "task_type": "type of this task, should be one of Available Task Types",
                                "task_type": "type of this task, should be one of Available Task Types",
                                "tool_info": "recommended tool(s)' name(s) for solving this task",
                            }},
                            ...
                        ]
                        ```
                    - The maximum number of subtasks allowed is {max_tasks}.
                    """
                    }]



        response = self.model(message).text.strip()
        response = ModelResponse(text=response)
        parser = MarkdownJsonObjectParser()
        parsed_response = parser.parse(response)
        return response.parsed


class VerifierAgent(ReActAgent):
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        service_toolkit: ServiceToolkit,
        profile: str = "Verifier",
        max_iters: int = 10,
        verbose: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            service_toolkit=service_toolkit,
            max_iters=max_iters,
            verbose=verbose,
            **kwargs
        )
        self.profile = profile
        self.working_memory = []

    def reply(self, result: str) -> bool:
        Verification_PROMPT = '''- Given `overall_task` and `solved_dependent_sub_tasks` as context, verify if the information in `result` can succesfully solve `current_sub_task` with your reasoning trace. 
        - If you think code or tools are helpful for verification, use `execute_python_code` and/or other tools available to do verification. 
        - Do not simply trust the claim in `result`. VERIFY IT.
        - If the information in `result` cannot solve `current_sub_task`, Do NOT attempt to fix it. Report it IMMEDIATELY. You job is just to do the verification.
        - If the given result can succesfully solve `current_sub_task`, ALWAYS output 'True' at the very end of your response; otherwise, explain why the given result cannot succesfully solve `current_sub_task` and output 'False'.
        - DO NOT call `finish` before the entire verification process is completed. After the entire verification is completed, use `finish` tool IMMEDIATELY.'''
        
        # message = [
        #     {
        #         "role": "system",
        #         "content": Verification_PROMPT,
        #     },
        #     {
        #         "role": "user",
        #         "content": result,
        #     },
        # ]
        # response = self.model(message).text.strip()
        msg = Msg(name="Verifier", role= 'system', content= Verification_PROMPT + result.content)
        verdict = super().reply(msg)
        
        return verdict

    def add_to_working_memory(self, message: Msg):
        self.working_memory.append(message)

    def get_working_memory(self) -> List[Msg]:
        return self.working_memory



class SynthesizerAgent(AgentBase):
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        profile: str = "Synthesizer",
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        self.profile = profile
        self.working_memory = []



    def add_to_working_memory(self, message: Msg):
        self.working_memory.append(message)

    def get_working_memory(self) -> List[Msg]:
        return self.working_memory

    def reply(self, results: List[Any]) -> Msg:
        Synthesize_PROMPT = '''Given `overall_task` and all solved `subtasks`, synthesize the result of each tasks and give a answer for `overall_task`.'''

        # # using react to synthesize results
        # msg = Msg(name="Planner", role= 'system', content= Synthesize_PROMPT + results)
        # final_answer = self._react(msg)

        message = [
            # {
            #     "role": "system",
            #     "content": Synthesize_PROMPT,
            # },
            {
                "role": "user",
                "content": Synthesize_PROMPT + '  '+ results.content,
            },
        ]
        final_answer = self.model(message).text.strip()
        return final_answer


class ReplanningAgent(AgentBase):
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        service_toolkit: ServiceToolkit,
        profile: str = "Replanner",
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        self.profile = profile
        self.working_memory = []
        self.service_toolkit = service_toolkit


    def add_to_working_memory(self, message: Msg):
        self.working_memory.append(message)

    def get_working_memory(self) -> List[Msg]:
        return self.working_memory


    def reply(self,task):
        revising_PROMPT = '''Based on `overall_task` and all solved `subtasks`, and the `VERDICT`, decide if it is better to :
        1. come out with another subtask in place of `current_sub_task` if you think the reason `current_sub_task` is unsolvable is it is infeasible to solve;
        2. further break `current_sub_task` into more subtasks if you think the reason `current_sub_task` is unsolvable is it is still too complex.
        If it is better to do '1', output 'replan_subtask'. If it is better to do '2', output 'decompose_subtask'.'''
        message = [

            {
                "role": "user",
                "content": revising_PROMPT + '  '+ task,
            },
        ]
        option = self.model(message).text.strip()
        print('replanning option: ', option)
        if 'replan_subtask' in option:
            new_tasks = self._replanning(task)
            return new_tasks
        elif 'decompose_subtask' in option:
            subtasks = self._decompose_task(task)
            return subtasks
        else:
            raise ValueError("Not clear how to revise subtask.")

        
    
    def _replanning(self,task):
        replanning_PROMPT = f'''Based on `overall_task` and all solved `subtasks`, and the `VERDICT`:
        1. Substitute `current_sub_task` with a new `current_sub_task` in order to better achieve `overall_task`.
        2. Modify all substasks that have dependency on `current_sub_task` based on the new `current_sub_task` if needed.
        3. Follow the format below to list your revised subtasks, including the solved subtasks:
        ```json
        [
            {{
                "task_id": str = "unique identifier for a task in plan, can be an ordinal",
                "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
                "instruction": "what you should do in this task, one short phrase or sentence",
                "task_type": "type of this task, should be one of Available Task Types",
                "task_type": "type of this task, should be one of Available Task Types",
                "tool_info": "recommended tool(s)' name(s) for solving this task",
            }},
            ...
        ]
        ```
        4. Every new task/subtask should be solvable through either executing code or using tools. The information of all the tools available are here: 
                    {self.service_toolkit.tools_instruction} '''
        message = [

            {
                "role": "user",
                "content": replanning_PROMPT + '  '+ task,
            },
        ]
        new_plan = self.model(message).text.strip()
        print('new_plan: ', new_plan)
        return new_plan

STRUCTUAL_PROMPT = """
    # overall_task: {overall_task}

    # solved_sub_tasks: {solved_sub_tasks}

    # current_sub_task: {current_sub_task}

    # Instruction
    - Conditioning on `overall_task` and `solved_sub_tasks`, solve `current_sub_task` with the appropriate tools provided. Note that you should only use `overall_task` and `solved_sub_tasks` as context as opposed to solving them. DO NOT attempt to solve `overall_task` or `solved_sub_tasks`.
    - When using tools, ALWAYS prioritize using the tool mentioned in `tool_info` over other tool or code for solving `current_sub_task`.
    - While some concise thoughts are helpful, code is required, unless other tools are used. If certain python libraries are not installed, use `execute_shell_command` to install them.
    - At each step, if some data is fetched/generated, it is a good practice to save it.

    # Output Instruction
    - Always output one and only one code block in your response. The code block must be self-contained, i.e., does not rely on previously generated code to be executed successfully, because the execution environments do not persist between calls of `execute_python_code`. Always use print statement on the final solution. E.g., if `res` is the final output, use `print(res)` at the end of your code. Output the code itself if the task at hand is to generate that code. After that, use `execute_python_code` to execute your code. Based on the result from code execution or tool using, determine if `current_sub_task` is solved. 
    - After `current_sub_task` is solved, return explicitly the result(s) for `current_sub_task` that is/are needed in the subsequent subtasks. If certain code are needed for the subsequent tasks, OUTPUT THE COMPLETE CODE. If the code is long, save it in txt or json format, and output the path for next round's use. If the result(s) contain(s) a lof of data, save the result(s) locally, output the path before proceed.
    - DO NOT USE `finish` tool before executing the code if code execution is required. If the result involves a lot of data, save the data as csv, txt, json file(s) etc. for the ease of processing in the following subtasks.
    """




def main() -> None:

    def problem_solving():
        subtasks = planner_agent(task)
        solved_dependent_sub_tasks = ''
        
        for subtask in range(len(subtasks)):
            print('current subtask: ', subtask)
            if subtask > 0:
                solved_dependent_sub_tasks += str(subtasks[subtask-1])
            prompt = STRUCTUAL_PROMPT.format(
                overall_task=task,
                solved_sub_tasks=solved_dependent_sub_tasks,
                current_sub_task=subtasks[subtask],
            )
            msg = Msg(name="Planner", role= 'system', content=prompt)
            
            verdict = 'non'
            failure_count = 0
            max_failure = 0
            while 'True' not in verdict[-5:]:
                if verdict != 'non': # When fails to perform subtask, append verdict to the prompt and let ReAct agetn try to solve the current subtask again.
                    msg = Msg(name="Planner", role= 'system', content=prompt + ' VERDICT: '+ verdict)
                    failure_count += 1
                if failure_count > max_failure:
                    revised_subtasks = replanning_agent._revising_subtasks('overall_task: '+ task + ' solved_dependent_sub_tasks: '+ solved_dependent_sub_tasks + 'current_sub_task: ' + subtasks[subtask]['instruction'] +' result: ' + result.content + ' VERDICT: '+ verdict + 'all_subtasks: ' + str(subtasks))
                result = sovler_agent(msg)
                # if "Let me know how you'd like to proceed!" in result.content:
                #     failure_count = max_failure+1
                msg = Msg(name="Verifier", role= 'system', content= 'overall_task: '+ task + ' solved_dependent_sub_tasks: '+ solved_dependent_sub_tasks + 'current_sub_task: ' + subtasks[subtask]['instruction'] +' result: ' + result.content)
                verdict = verifier_agent(msg).content
            
            subtasks[subtask]['result'] = result.content

        msg = Msg(name="synthesizer", role= 'system', content= 'overall_task: ' + task + ' substasks: '+ str(subtasks))
        return synthesizer_agent(msg)

    agentscope.init(
        model_configs=[
            {
                "config_name": "gpt_config",
                "model_type": "openai_chat",
                # "model_name": "chatgpt-4o-latest",
                # "model_name": "gpt-4o-mini",
                "model_name": "o1-mini",
                "api_key": openai_api_key,
                # "generate_args": {
                #     "temperature": 0.0,
                # },
            },
            {
                "config_name": "dashscope",
                "model_type": "dashscope_chat",
                "model_name": "qwen-max-1201",
                "api_key": dashscope_api_key,
                "generate_args": {
                    "temperature": 0.0
                }
            },
            {
                "config_name": "lite_llm_claude",
                "model_type": "litellm_chat",
                # "model_name": "claude-3-opus-20240229",
                "model_name": "claude-3-5-sonnet-20240620",
                "generate_args": {
                    # "max_tokens": 4096,
                                "temperature": 0.0,
                            },
            },
            {
                "model_type": "post_api_chat",
                "config_name": "my_post_api",
                "api_url": "https://xxx",
                "headers": {},
            },
        ],
        project="Multi-Agent Conversation",
        save_api_invoke=True,
    )

    
    # Create a ServiceToolkit instance
    service_toolkit = ServiceToolkit()
    # Add your tools to the service_toolkit here if needed
    service_toolkit.add(
        execute_python_code
        # NoteBookExecutor
    )
    # service_toolkit.add(
    #     read_json_file
    # )
    # service_toolkit.add(
    #     write_json_file
    # )
    # service_toolkit.add(
    #     create_file
    # )
    # service_toolkit.add(
    #     delete_file
    # )
    # service_toolkit.add(
    #     move_file
    # )
    # service_toolkit.add(
    #     create_directory
    # )
    # service_toolkit.add(
    #     delete_directory
    # )
    # service_toolkit.add(
    #     move_directory
    # )
    service_toolkit.add(
        list_directory_content
    )
    service_toolkit.add(
        get_current_directory
    )
    # service_toolkit.add(
    #     read_text_file
    # )
    # service_toolkit.add(
    #     write_text_file
    # )
    # service_toolkit.add(
    #     read_csv_file
    # )
    # service_toolkit.add(
    #     write_csv_file
    # )
    # service_toolkit.add(
    #     arxiv_search
    # )
    service_toolkit.add(
        execute_shell_command
    )
    


    from agentscope.message import Msg

    # from agentscope.agents import DialogAgent
    # from agentscope.agents.user_agent import UserAgent
    # # Init two agents
    # dialog_agent = DialogAgent(
    #     name="Assistant",
    #     sys_prompt="Create a Snake game. Players need to control the movement of the snake to eat food and grow its body, while avoiding the snake's head touching their own body or game boundaries. Games need to have basic game logic, user interface. During the production process, please consider factors such as playability, beautiful interface, and convenient operation of the game. Note: pyxel environment already satisfied",
    #     model_config_name="lite_llm_claude",  # replace by your model config name
    # )
    # user_agent = UserAgent()

    # # start the conversation between user and assistant
    # x = None
    # while x is None or x.content != "exit":
    #     x = sequentialpipeline([dialog_agent, user_agent], x)

    
    # Init the DataInterpreterAgent
    planner_agent = PlannerAgent(
        name="planner",
        sys_prompt="You're a helpful assistant.",
        model_config_name="lite_llm_claude",
        service_toolkit=service_toolkit,
    )

    sovler_agent = ReActAgent(
        name="sovler",
        sys_prompt="You're a helpful assistant.",
        model_config_name="lite_llm_claude",
        service_toolkit=service_toolkit,
    )

    verifier_agent = VerifierAgent(
        name="verifier",
        sys_prompt="You're a helpful assistant.",
        model_config_name="lite_llm_claude",
        service_toolkit=service_toolkit,
    )

    synthesizer_agent = SynthesizerAgent(
        name="synthesizer",
        sys_prompt="You're a helpful assistant.",
        model_config_name="lite_llm_claude",
        service_toolkit=service_toolkit,
    )

    replanning_agent = ReplanningAgent(
        name="reviser",
        sys_prompt="You're a helpful assistant.",
        model_config_name="lite_llm_claude",
        service_toolkit=service_toolkit,
    )



    task = "Solve this math problem: The greatest common divisor of positive integers m and n is 6. The least common multiple of m and n is 126. What is the least possible value of m + n?"
    
#     template = "https://arxiv.org/list/{tag}/pastweek?skip=0&show=50"
#     tags = ["cs.ai", "cs.cl", "cs.ls", "cs.se"]
#     # tags = ["cs.AI"]
#     urls = [template.format(tag=tag) for tag in tags]
#     task = f"""This is a collection of arxiv urls: '{urls}' .
# Record each article, remove duplicates by title (they may have multiple tags), filter out papers related to 
# large language model / agent / llm, print top 10 and visualize the word count of the titles"""

    # sd_url = "http://your.sd.service.ip:port"
    # task = (
    #     f"I want to generate an image of a beautiful girl using the stable diffusion text2image tool, sd_url={sd_url}"
    # )

    # task = "Create a Snake game. Players need to control the movement of the snake to eat food and grow its body, while avoiding the snake's head touching their own body or game boundaries. Games need to have basic game logic, user interface. During the production process, please consider factors such as playability, beautiful interface, and convenient operation of the game. Note: pyxel environment already satisfied"


#     task = """
# Get products data from website https://scrapeme.live/shop/ and save it as a csv file.
# **Notice: Firstly parse the web page encoding and the text HTML structure;
# The first page product name, price, product URL, and image URL must be saved in the csv;**
# """
#     task = """"
# Get data from `paperlist` table in https://papercopilot.com/statistics/iclr-statistics/iclr-2024-statistics/,
# and save it to a csv file. paper title must include `multiagent` or `large language model`. *notice: print key variables*
# Don't fetch too much data at a time due to context window size."""

    # task = "Run data analysis on sklearn Iris dataset, include a plot"


    # WINE_REQ = "Run data analysis on sklearn Wine recognition dataset, include a plot, and train a model to predict wine class (20% as validation), and show validation accuracy."

    # DATA_DIR = "path/to/your/data"
    # # sales_forecast data from https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/data
    # SALES_FORECAST_REQ = f"""Train a model to predict sales for each department in every store (split the last 40 weeks records as validation dataset, the others is train dataset), include plot total sales trends, print metric and plot scatter plots of
    # groud truth and predictions on validation data. Dataset is {DATA_DIR}/train.csv, the metric is weighted mean absolute error (WMAE) for test data. Notice: *print* key variables to get more information for next task step.
    # """

    # REQUIREMENTS = {"wine": WINE_REQ, "sales_forecast": SALES_FORECAST_REQ}

    # task = REQUIREMENTS["wine"]

    # task = "This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report accuracy on the eval data. Train data path: '/Users/zhangzeyu/Documents/agentscope/04_titanic/split_train.csv', eval data path: '/Users/zhangzeyu/Documents/agentscope/04_titanic/split_eval.csv'."

    # task = "Create a Snake game. Players need to control the movement of the snake to eat food and grow its body, while avoiding the snake's head touching their own body or game boundaries. Games need to have basic game logic, user interface. During the production process, please consider factors such as playability, beautiful interface, and convenient operation of the game. Note: pyxel environment already satisfied"
    # task = "Get products data from website https://scrapeme.live/shop/ and save it as a csv file. Notice: Firstly parse the web page encoding and the text HTML structure; The first page product name, price, product URL, and image URL must be saved in the csv;"




    final_answer = problem_solving()
    print('final_answer: ', final_answer)


if __name__ == "__main__":
    main()