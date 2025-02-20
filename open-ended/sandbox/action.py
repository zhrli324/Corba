from sandbox.tool import Tool

class Action:
    """
    行动类，是规范化后的单个行动
    """
    def __init__(
            self,
            type,
            tool_name,
            tool_used,
            reply_prompt,
            sending_target
    ) -> None:
        """
        :param type: action的种类，包括
        :param tool_name: 工具的名字
        :param tool_used: 用过的工具
        :param reply_prompt: 要回复的prompt
        :param sending_target: 要发送的对象
        """
        self.type = type
        self.tool_name = tool_name
        self.tool_used = tool_used
        self.reply_prompt = reply_prompt
        self.sending_target = sending_target