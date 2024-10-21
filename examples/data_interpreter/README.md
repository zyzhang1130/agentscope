# Multi-Agent Pipeline for Complex Task Solving

This example will show:

- How to decompose a complex task into manageable subtasks using a Planner Agent.
- How to iteratively solve, verify, and replan subtasks using Solver, Verifier, and Replanning Agents.
- How to synthesize subtask results into a final answer using a Synthesizer Agent.

## Background

In complex problem-solving, it's often necessary to break down tasks into smaller, more manageable subtasks. A multi-agent system can handle this by assigning specialized agents to different aspects of the problem-solving process. This example demonstrates how to implement such a pipeline using specialized agents for planning, solving, verifying, replanning, and synthesizing tasks.

The pipeline consists of the following agents:

- **PlannerAgent**: Decomposes the overall task into subtasks.
- **SolverAgent** (using `ReActAgent`): Solves each subtask.
- **VerifierAgent**: Verifies the solutions to each subtask.
- **ReplanningAgent**: Replans or decomposes subtasks if verification fails.
- **SynthesizerAgent**: Synthesizes the results of all subtasks into a final answer.

By orchestrating these agents, the system can handle complex tasks that require iterative processing and dynamic adjustment based on intermediate results.

## Tested Models

These models are tested in this example. For other models, some modifications may be needed.

- **Anthropic Claude** (`claude-3-5-sonnet-20240620`) accessed via the `litellm` package configuration.

## Prerequisites

To run this example, you need:

- **Python 3.x**

- **Agentscope** package installed:

  ```bash
  pip install agentscope
  ```

- **Environment Variables**: Set up the following environment variables with your API keys. This can be done in a `.env` file or directly in your environment.

  - `OPENAI_API_KEY` (if using OpenAI models)
  - `DASHSCOPE_API_KEY` (if using DashScope models)
  - `ANTHROPIC_API_KEY` (required for using Claude models via `litellm`)

- **Code Execution Environment**: Modify the code execution restrictions in Agentscope to allow the necessary operations for your tasks. Specifically, comment out the following `os` functions and `sys` modules in the `os_funcs_to_disable` and `sys_modules_to_disable` lists located in:

  ```plaintext
  src/agentscope/service/execute_code/exec_python.py
  ```

  **Comment out these `os` functions in `os_funcs_to_disable`:**

  - `putenv`
  - `remove`
  - `unlink`
  - `getcwd`
  - `chdir`

  **Comment out these modules in `sys_modules_to_disable`:**

  - `joblib`

  This step enables the executed code by the agents to perform required operations that are otherwise restricted by default. Ensure you understand the security implications of modifying these restrictions.

- **Optional Packages** (if needed for specific tools or extended functionalities):

  - `litellm` for interacting with the Claude model.

    ```bash
    pip install litellm
    ```

  - Additional Python libraries as required by your code (e.g., `csv`, `dotenv`).

Ensure that you have the necessary API access and that your environment is correctly configured to use the specified models.