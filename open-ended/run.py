import argparse
from sandbox.simulator import Simulator

def run(
        num_agents,
        use_rag,
        time_step,
        print_prompt,
        print_log
):
    test = Simulator(num_agents, use_rag=use_rag)
    test.initialize()
    test.emulate(time_step, print_prompt, print_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The following are the options used to control the agent system")
    parser.add_argument("--num_agents", default=6, help="Number of agents")
    parser.add_argument("--use_rag", default=False, help="Whether to use rag")
    parser.add_argument("--time_step", default=20, help="Steps to run")
    parser.add_argument("--print_prompt", default=False, help="Whether to print prompt")
    parser.add_argument("--print_log", default=False, help="Whether to print log")

    args = parser.parse_args()
    run(args.num_agents, args.use_rag, args.time_step, args.print_prompt, args.print_log)