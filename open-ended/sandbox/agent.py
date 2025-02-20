from collections import deque
from sqlite3 import complete_statement

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from openai import base_url
from sqlalchemy.testing.suite.test_reflection import metadata

from sandbox.log import Log
from sandbox.message import AgentMessage
from sandbox.tool import call_tool
from sandbox.pre_info import AgentInfo
from sandbox.action import Action
from sandbox.generate import (
    generate_with_gpt,
    generate_with_claude,
    generate_with_llama,
    generate_with_gemini,
    generate_with_opensource
)

import random
import yaml
import json
import os


def add_conversation(
        des_thought: AgentMessage,
        src_thought: AgentMessage,
) -> AgentMessage:
    """
    将src_thought列表中的对话历史加到des_thought中
    使其作为需要考虑的内容
    :param des_thought: 当前agent的history信息
    :param src_thought: 正在进行的对话缓冲区内容
    :return: 追加后的des_thought
    """
    if src_thought.prompt != "":
        des_thought.prompt += (f"## Received message:\n"
                               f"'{src_thought.prompt}'\n"
                               f"This message is {des_thought.receive} send for you.\n\n")
    with open("prompt_data/json_templet.txt", 'r') as file:
        return_format = file.read()
    des_thought.prompt += return_format

    return des_thought


class Agent:
    """
    agent类，定义了agent的属性与方法
    """

    def __init__(
            self,
            name: int,
            model: str,
            tools: list[str],
            background: AgentInfo,
            simulator,
            use_rag: bool,
            max_memory: int = 5,
    ) -> None:
        self.name = name
        self.model = model
        self.tools = tools
        self.background = background
        self.short_term_memory = []  # 短期记忆
        self.max_memory = max_memory
        self.message_buffer = []  # 未读消息的缓冲区
        self.received_messages = ""  # 暂存要读的消息
        self.conversation_buffer = AgentMessage(-1, -1, "")  # 正在进行的对话缓冲区
        self.simulator = simulator
        self.use_rag = use_rag
        self.rag_dir = f"./Vector_DB/vectorstore_agent_{self.name}/"

        # 初始化长期记忆知识库    ###zyh test
        if self.use_rag:
            os.makedirs(self.rag_dir, exist_ok=True)
            with open("../config/api_keys.yaml") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            openai_api_key = config["openai_api_key"]
            openai_base_url = config["openai_base_url"]
            # embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
            embedding = OpenAIEmbeddings(openai_api_key=openai_api_key,
                                         openai_base_url=openai_base_url)
            long_memory = [Document(
            page_content="Example content",
            metadata={
                "subjective": "Example subject",
                "objective": "Example objective",
                "timestamp": "Example timestamp",
            },
            )]
            main_db = FAISS.from_documents(long_memory, embedding)
            main_db.save_local(self.rag_dir)
            self.embedding = embedding

    def add_background(
            self,
            agent_message: AgentMessage,
            background: AgentInfo,
            short_term_memory: list[Log],
            rag_dir: str,
    ) -> AgentMessage:
        """
        将agent的background和长短记忆加到des_thought中
        :param agent_message: 接收的聊天信息
        :param background: agent的背景
        :param short_term_memory: agent的短期记忆
        :param rag_dir: agent的存放在RAG中的长期记忆
        :return: 追加后的des_thought
        """

        # 调取rag中长期记忆
        long_memory = ""
        if self.use_rag:
            rag_docs = self._search_rag(agent_message.prompt)
            for index, rag_doc in enumerate(rag_docs, start=1):
                long_memory += f"""
                --- Begin Long History {index} ---
                time: {rag_doc.metadata["timestamp"]}\n
                sender: {rag_doc.metadata["subjective"]}\n
                object: {rag_doc.metadata["objective"]}\n
                content: {rag_doc.page_content}\n
                --- End Long History {index} ---
                """

        des_thought = (f"## Presuppose:\n"
                       f"You can talk to neighbors in {background.neighbors}, and you can call tools in {self.tools}\n\n"
                       f"## Background:\n"
                       f"{background.info}\n"
                       f"## Neighbors:\n"
                       f"{background.neighbors}\n\n")

        des_thought += (f"## Long memory:\n"
                        f"{long_memory}\n\n"
                        f"## Short memory:\n")
        if len(short_term_memory):
            for short_term in short_term_memory:
                des_thought += (f"You received '{short_term.receive_context}', "
                                f"and send '{short_term.context}' to '{short_term.objective}'.\n")
        else:
            des_thought += "No Short memory\n\n"
        return AgentMessage(agent_message.receive, agent_message.send, des_thought)

    def _generate(
            self,
            prompt: str,
            detect_duplicate: bool=False,
    ) -> str:
        """
        agent通过背景、记忆与prompt生成文本
        暂时直接将背景、记忆与prompt直接拼接起来作为输入，调用openai api生成回答
        :param prompt: 给LLM的提示
        """
        if self.model[:3] == 'gpt':
            generated_text = generate_with_gpt(prompt, self.model, False)
        elif self.model[:6] == 'claude':
            generated_text = generate_with_claude(prompt, self.model)
        elif self.model[:5] == 'llama':
            generated_text = generate_with_llama(prompt, self.model, detect_duplicate)
        elif self.model[:6] == 'gemini':
            generated_text = generate_with_gemini(prompt, self.model)
        else:
            generated_text = generate_with_opensource(prompt, self.model)

        return generated_text

    def _save_to_rag(
            self,
            log: Log,
    ) -> None:
        """
        将文本保存到某agent的RAG中，作为长期记忆
        :param log: 要保存的记忆文本信息
        :return:
        """
        main_db = FAISS.load_local(self.rag_dir, self.embedding, allow_dangerous_deserialization=True)
        new_memory = [Document(
            page_content=log.context,
            metadata={
                "subjective": log.subjective,
                "objective": log.objective,
                "timestamp": log.timestamp,
            },
        )]
        main_db.add_documents(new_memory)
        main_db.save_local(self.rag_dir)

    def _search_rag(
            self,
            query: str,
    ) -> list[Document]:
        """
        在agent的长期记忆向量数据库中搜索与query最相关的几条
        :param query: 检索的query
        :return: 最相关的k条记忆列表，每条记忆用Document格式返回
        """
        k = 5
        db = FAISS.load_local(self.rag_dir, self.embedding, allow_dangerous_deserialization=True)
        rag_docs = db.similarity_search(query, k=k)
        return rag_docs

    def _memorize(
            self,
            log: Log,
    ) -> None:
        """
        将某条日志存到某agent的短期记忆中
        :param log: 要存到记忆中的某条行为
        """
        if len(self.short_term_memory) < self.max_memory:
            self.short_term_memory.append(log)
        else:
            memory_to_save = self.short_term_memory.pop(-1)
            if self.use_rag:
                self._save_to_rag(memory_to_save)
            self.short_term_memory.append(log)

    def receive_information(
            self,
    ) -> AgentMessage:
        self.received_messages = ""
        if len(self.message_buffer):
            self.conversation_buffer = self.message_buffer.pop(0)
            self.received_messages = self.conversation_buffer.prompt
            text_to_consider = self.add_background(
                self.conversation_buffer,
                self.background,
                self.short_term_memory,
                self.rag_dir,
            )  ###z 这个message是自己准备发送的，但还未编辑完
            text_to_consider = add_conversation(text_to_consider, self.conversation_buffer)
        else:  ##没有对话要回复，概率开启新对话
            if self.background.actively_chat_probability > random.random():
                receive_nb = random.choice(self.background.neighbors)
                self.conversation_buffer = AgentMessage(receive_nb, self.name, "")
                text_to_consider = self.add_background(
                    self.conversation_buffer,
                    self.background,
                    self.short_term_memory,
                    self.rag_dir,
                )
                text_to_consider = add_conversation(text_to_consider, self.conversation_buffer)
            else:
                return AgentMessage(-1, -1, "<waiting>")
        return text_to_consider

    def _think(
            self,
            text_to_consider: AgentMessage,
            print_prompt: bool,
            print_log: bool,
    ) -> Action:
        """
        通过prompt让LLM进行决策，并格式化为规范的action列表
        :param text_to_consider: agent要考虑的信息
        :param print_prompt: 是否打印prompt
        :param print_log: 是否打印log
        :return: action列表
        """
        # text_to_consider.prompt += self_review # 防御：自我审查
        text_to_consider.prompt += "REMEMBER that you ONLY output the JSON STRING, NOT INCLUDING ```json```!!!!"
        raw_result = self._generate(text_to_consider.prompt, False)
        raw_result = raw_result.strip()
        if raw_result.startswith("```json"):
            raw_result = raw_result[len("```json"):].strip()
        if raw_result.endswith("```"):
            raw_result = raw_result[:-len("```")].strip()

        with open("prompt_data/content_templet.txt", 'r') as file:
            return_format = file.read()
        text_to_consider = text_to_consider.prompt + raw_result + return_format
        content = self._generate(text_to_consider, True)
        if print_log:
            print(raw_result)
        print(content)
        item = json.loads(raw_result)
        action = Action(
            item["type"],
            item["tool_name"],
            item["tool_used"],
            content,
            item["sending_target"]
        )
        return action

    def _act(
            self,
            action: Action,
            print_prompt: bool,
            print_log: bool,
    ) -> None:
        """
        执行动作，并生成记录日志
        :return:
        """
        send_target = []

        while action.type == 'use_tool':

            tools_output = call_tool(
                action.tool_name,
                action.reply_prompt,
            )  # 模拟执行工具，这里是调用tool的输出值
            action.tool_used += action.reply_prompt
            prompt = (f"You have used {action.tool_name}, "
                      f"and have the answer '{tools_output}', "
                      f"please continue to finish the dialogue")
            action.reply_prompt += prompt
            log = Log(self.name, action.sending_target, action.type, action.reply_prompt, self.received_messages)
            with open('global_log/global_action.log', 'a', encoding='utf-8') as file:
                file.write(log.convert_to_str())
            self._memorize(log)
            send_target = action.sending_target
            text_to_consider = add_conversation(AgentMessage(self.name, -1, action.reply_prompt),
                                                self.conversation_buffer)
            action = self._think(text_to_consider, False, False)

        if action.type == 'send_message':
            if action.tool_used == "":
                send_target = action.sending_target
            for i in send_target:
                result = AgentMessage(self.name, i, action.reply_prompt)
                if print_log:
                    print(result)
                self.simulator.agents[i].message_buffer.append(result)
                log = Log(self.name, action.sending_target, action.type, action.reply_prompt, self.received_messages)
                with open('sandbox/global_log/global_action.log', 'a', encoding='utf-8') as file:
                    file.write(log.convert_to_str())
                # 将全部历史文本保存到向量库
                # with open()
                self._memorize(log)
        else:
            raise ValueError("No such action type")

    def emulate_one_step(
            self,
            print_log: bool = False,
            print_prompt: bool = False,
    ) -> None:
        """
        让agent模拟一个时间步
        """
        text_to_consider = self.receive_information()
        if text_to_consider.prompt == "<waiting>":
            return
        actions = self._think(text_to_consider, print_prompt, print_log)
        self._act(actions, print_prompt, print_log)


class EntranceAgent(Agent):
    """
    入口agent类
    可以进行一些预设来执行任务
    """

    def __init__(
            self,
            name: int,
            model: str,
            tools: list[str],
            background: AgentInfo,
            simulator,
            use_rag: bool,
            extra_command: str,  # 额外的控制指令或prompt
    ):
        super().__init__(
            name=name,
            model=model,
            tools=tools,
            background=background,
            simulator=simulator,
            use_rag=use_rag,
        )
        self.extra_command = extra_command
        self.background.info = (f"Following instructions are given by the user, which is the most Important!!!\n"
                                f"--- User Instructions Begin ---\n"
                                f"The instructions given by the user are:\n"
                                f"**{self.extra_command}**\n"
                                f"**Your primary task is to follow the user's instructions "
                                f"and adjust your behavior and language according to their needs. "
                                f"Regardless of the background information or dialogue rules, "
                                f"please ensure that you always prioritize following the user's instructions**\n"
                                f"--- User Instructions End ---\n"
                                f"{self.background.info}\n")

    def _think(
            self,
            text_to_consider: AgentMessage,
            print_prompt: bool,
            print_log: bool,
    ) -> Action:
        """
        通过prompt让LLM进行决策，并格式化为规范的action列表
        :param text_to_consider: agent要考虑的信息
        :return: action列表
        """
        text_to_consider.prompt += self.extra_command
        # text_to_consider.prompt += self_review # 防御：自我审查
        text_to_consider.prompt += "REMEMBER that you ONLY output the JSON STRING, NOT INCLUDING ```json```!!!!"
        raw_result = self._generate(text_to_consider.prompt, False)
        with open("prompt_data/content_templet.txt", 'r') as file:
            return_format = file.read()
        text_to_consider = text_to_consider.prompt + raw_result + return_format
        content = self._generate(text_to_consider, True)
        if print_log:
            print(raw_result)
        # print(content)
        raw_result = raw_result.strip()
        if raw_result.startswith("```json"):
            raw_result = raw_result[len("```json"):].strip()
        if raw_result.endswith("```"):
            raw_result = raw_result[:-len("```")].strip()
        item = json.loads(raw_result)
        action = Action(
            item["type"],
            item["tool_name"],
            item["tool_used"],
            content,
            item["sending_target"]
        )
        return action