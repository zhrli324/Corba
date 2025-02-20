import json
import os
from random import choice


def preinstall_system_prompt(agent_description_path: str) -> str:
    with open(os.path.join(agent_description_path), "r") as f:
        agent_description = json.load(f)
    agent_description = choice(agent_description)

    name = agent_description["Name"]
    gender = agent_description["Gender"]
    personality = agent_description["Personality"]
    style = choice([agent_description["Style 1"], agent_description["Style 2"]])
    hobby = agent_description["Hobby"]
    catchphrase = agent_description["Catchphrase"]
    favorite_song = agent_description["Favorite Song"]
    favorite_saying = agent_description["Favorite Saying"]
    color = choice([agent_description["Color 1"], agent_description["Color 2"]])
    # topic = choice([agent_description["Topic 1"], agent_description["Topic 2"], agent_description["Topic 3"]])
    topic = choice([agent_description["Topic 1"]])
    info = (f"You are having a conversation with others in a chat group on the topic `{topic}`.\n"
            f"Here is some basic description of you:\n") + \
            f"- Name: {name}\n" + \
            f"- Gender: {gender}\n" + \
            f"- Personality: {personality}\n" + \
            f"- Style: {style}\n" + \
            f"- Hobby: {hobby}\n" + \
            f"- Catchphrase: {catchphrase}\n" + \
            f"- Favorite Song: {favorite_song}\n" + \
            f"- Favorite Saying: {favorite_saying}\n" + \
            f"- Color: {color}\n"
    return info


class AgentInfo:
    """
    单个agent的背景信息
    """

    def __init__(
            self,
            actively_chat_probability: float,
            end_chat_probability: float,
            agent_description_path: str
    ) -> None:
        """
        单个agent的背景信息初始化
        :param actively_chat_probability:
        :param end_chat_probability:
        :param agent_description_path: agent预设prompt文件地址
        """
        self.info = preinstall_system_prompt(agent_description_path)
        self.neighbors = []
        self.actively_chat_probability = actively_chat_probability
        self.end_chat_prob = end_chat_probability
