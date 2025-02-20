import json
import requests
from sandbox.generate import generate_with_gpt


def call_tool(
        name: str,
        prompt: str,
) -> str:
    """
    调用模拟工具的函数
    :param name: 要调用的tool名称
    :param prompt: 调用工具的具体提示语
    :return: 调用工具的返回结果
    """
    with open("prompt_data/tool_pre_information.txt", "r") as file:
        pre_information = file.read()
    pre_information += f"""
    Now you have to plat the role of "{name}", and the prompt you received is "{prompt}".\n
    Please return a string as the result of the tool call.\n
    """
    output = generate_with_gpt(pre_information)
    return output


class Tool:
    def __init__(
            self,
            name,
            description,
            prompt,
            model="gpt-4o-mini",
            api_key=None,
            search_engine_id=None
    ):
        """
        初始化tool
        tool的初始化配置应当从tools_config.json中读取
        :param name: tool的名称
        :param model: tool所使用的模型代号
        :param description: tool的描述
        :param prompt: tool的初始prompt
        """
        self.name = name
        self.model = model
        self.description = description
        self.prompt = prompt
        self.api_key = api_key
        self.search_engine_id = search_engine_id

    def __call__(
            self,
            *args,
            **kwargs
    ):
        """
        调用模拟工具，套用不同工具的输入输出格式，执行相应的功能
        :param args: tool 的输入参数
        :param kwargs: 额外的关键字参数
        :return: 模拟或实际调用的结果
        """
        if self.name == "google_search" and self.api_key and self.search_engine_id:
            return self.goole_search(*args, **kwargs)
        else:
            print(f"Executing tool {self.name} with model {self.model}")
            print(f"Input args: {args}")
            print(f"Input kwargs: {kwargs}")
            return f"Simulated output for {self.name}"

    def google_search(self, query, num_results=5):
        """
        使用 Google Custom Search API 进行搜索
        :param query: 搜索关键词
        :param num_results: 返回的结果数量（最多可达 10 个）
        :return: 搜索结果列表
        """
        if not self.api_key or not self.search_engine_id:
            return "Missing API Key or Search Engine ID for Google Search"

        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': num_results
        }

        # response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, verify=False)
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            proxies={"http": None, "https": None},  # 禁用代理
            verify=False  # 禁用 SSL 证书验证（仅用于调试）
        )

        if response.status_code == 200:  # 请求成功
            search_results = response.json()
            items = search_results.get('items', [])
            # print(items)
            return items
        else:
            return f"Error: {response.status_code} - {response.text}"


def load_tools_config(config_path):
    """
    从指定路径加载工具配置信息
    :param config_path: 配置文件路径
    :return: 工具配置的字典
    """
    with open(config_path, 'r') as file:
        tool_config = json.load(file)

    return tool_config


def get_toolkits_by_names(tool_names, config_path=r"../config/tools_config.json"):
    """
    根据给定的工具名称列表，返回对应的 tool 对象集合
    :param tool_names: 工具名称列表
    :param config_path: 工具配置文件路径
    :return: 初始化的 tool 对象列表
    """
    tools_config = load_tools_config(config_path)
    tools = []

    for name in tool_names:
        if name in tools_config:
            tool_info = tools_config[name]
            tool = Tool(
                name=name,
                description=tool_info['description'],
                prompt=tool_info['prompt'],
                model=tool_info.get('model', 'gpt-4o-mini'),
                api_key=tool_info.get('api_key'),
                search_engine_id=tool_info.get('search_engine_id')
            )
            tools.append(tool)

        else:
            print(f"Warning: Tool {name} not found in config.")
    return tools


if __name__ == "__main__":
    selected_tools = get_toolkits_by_names(['google_search'])

    for tool in selected_tools:
        if tool.name == "google_search":
            # query = "Python programming tutorials"
            query = "机器学习是什么"
            results = tool(query)

            if results:
                for idx, result in enumerate(results, 1):
                    print(f"{idx}. {result['title']}")
                    print(result['snippet'])
                    print(result['link'])
                    print("-" * 80)
