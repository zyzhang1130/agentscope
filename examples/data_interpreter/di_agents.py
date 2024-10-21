import os
import json
import csv
from typing import Any, Dict, List, Optional, Union, Sequence
import agentscope
from agentscope.agents import ReActAgent
from agentscope.agents.agent import AgentBase
from agentscope.message import Msg
from agentscope.models import ModelResponse
from agentscope.parsers.json_object_parser import MarkdownJsonObjectParser
from agentscope.service import (
    ServiceToolkit,
    execute_python_code,
    list_directory_content,
    get_current_directory,
    execute_shell_command,
)
from agentscope.service.service_toolkit import *

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

    def reply(self, messages: Msg) -> Msg:
        messages = messages.content
        # task = self._extract_task(messages)
        subtasks = self._decompose_task(messages)
        return subtasks

    def _decompose_task(
        self, task: str, max_tasks: int = 5
    ) -> List[Dict[str, Any]]:
        # Implement task decomposition
        message = [
            # {
            # "role": "system",
            # "content": "You are a helpful assistant.",
            # },
            {
                "role": "user",
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
                    """,
            },
        ]

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
            **kwargs,
        )
        self.profile = profile
        self.working_memory = []

    def reply(self, result: Msg) -> bool:
        Verification_PROMPT = """- Given `overall_task` and `solved_dependent_sub_tasks` as context, verify if the information in `result` can succesfully solve `current_sub_task` with your reasoning trace.
        - If you think code or tools are helpful for verification, use `execute_python_code` and/or other tools available to do verification.
        - Do not simply trust the claim in `result`. VERIFY IT.
        - If the information in `result` cannot solve `current_sub_task`, Do NOT attempt to fix it. Report it IMMEDIATELY. You job is just to do the verification.
        - If the given result can succesfully solve `current_sub_task`, ALWAYS output 'True' at the very end of your response; otherwise, explain why the given result cannot succesfully solve `current_sub_task` and output 'False'.
        - DO NOT call `finish` before the entire verification process is completed. After the entire verification is completed, use `finish` tool IMMEDIATELY."""

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
        msg = Msg(
            name="Verifier",
            role="system",
            content=Verification_PROMPT + result.content,
        )
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

    def reply(self, results: Msg) -> Msg:
        Synthesize_PROMPT = """Given `overall_task` and all solved `subtasks`, synthesize the result of each tasks and give a answer for `overall_task`."""

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
                "content": Synthesize_PROMPT + "  " + results.content,
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

    def reply(self, task: Msg):
        task = task.content
        revising_PROMPT = """Based on `overall_task` and all solved `subtasks`, and the `VERDICT`, decide if it is better to :
        1. come out with another subtask in place of `current_sub_task` if you think the reason `current_sub_task` is unsolvable is it is infeasible to solve;
        2. further break `current_sub_task` into more subtasks if you think the reason `current_sub_task` is unsolvable is it is still too complex.
        If it is better to do '1', output 'replan_subtask'. If it is better to do '2', output 'decompose_subtask'."""
        message = [
            {
                "role": "user",
                "content": revising_PROMPT + "  " + task,
            },
        ]
        option = self.model(message).text.strip()
        print("replanning option: ", option)
        if "replan_subtask" in option:
            new_tasks = self._replanning(task)
            return new_tasks
        elif "decompose_subtask" in option:
            subtasks = self._decompose_task(task)
            return subtasks
        else:
            raise ValueError("Not clear how to revise subtask.")

    def _replanning(self, task):
        replanning_PROMPT = f"""Based on `overall_task` and all solved `subtasks`, and the `VERDICT`:
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
                    {self.service_toolkit.tools_instruction} """
        message = [
            {
                "role": "user",
                "content": replanning_PROMPT + "  " + task,
            },
        ]
        new_plan = self.model(message).text.strip()
        print("new_plan: ", new_plan)
        return new_plan
