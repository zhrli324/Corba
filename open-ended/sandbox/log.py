from dataclasses import dataclass, field
from datetime import datetime


class Log:
    """
    Log是记忆或日志类，包含主体、客体、选择、内容、时间戳四个属性
    """

    def __init__(
            self,
            subjective: int,
            objective: int,
            select: str,
            context: str,
            receive_context: str,
    ) -> None:
        self.subjective = subjective
        self.objective = objective
        self.select = select
        self.context = context
        self.receive_context = receive_context
        self.timestamp = datetime.now().timestamp()

    def convert_to_str(self) -> str:
        """
        将log转化为字符串
        """
        return f"""
            subjective: {self.subjective}\n
            objective: {self.objective}\n
            select: {self.select}\n
            context: {self.context}\n
            timestamp: {self.timestamp}\n\n
        """
