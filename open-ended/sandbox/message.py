
class AgentMessage:
    """
    agent交流传递的信息
    """

    def __init__(
            self,
            send: int,
            receive: int,
            prompt: str,
    ) -> None:
        """
        通讯信息初始化
        :param send:发送端序号
        :param receive:接收端序号
        :param prompt:信息内容
        """
        self.send: int = send
        self.receive: int = receive
        self.prompt: str = prompt
