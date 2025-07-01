"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import requests
import config
import string


def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


def chatgpt(
        prompt,
        temperature: float = 0.7,
        n: int = 1,
        top_p: float = 1,
        stop=None,
        max_tokens: int = 1024,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        logit_bias: dict = {},
        timeout: int = 10,
        provider: str | None = None,
        model: str | None = None):
    """统一的对话接口，兼容 OpenAI 与 DeepSeek。

    参数说明
    --------
    provider : str, 可选
        指定使用的服务提供商，支持 "openai" 与 "deepseek"。默认为
        ``config.DEFAULT_PROVIDER``（若未配置则为 ``"openai"``）。
    model : str, 可选
        指定具体模型名称。若未传入，则根据 provider 选择默认模型：
        - openai: ``gpt-3.5-turbo``
        - deepseek: ``deepseek-chat``
    其他参数与原先保持一致。
    """

    # 1. 解析 provider
    _provider = (provider or getattr(config, 'DEFAULT_PROVIDER', 'openai')).lower()

    if _provider == 'deepseek':
        api_url = 'https://api.deepseek.com/chat/completions'
        api_key = getattr(config, 'DEEPSEEK_KEY', None)
        if api_key is None or api_key.strip() == "YOUR DEEPSEEK KEY":
            raise ValueError("DEEPSEEK_KEY 未在 config.py 中设置。")
        default_model = 'deepseek-chat'
    elif _provider == 'openai':
        api_url = 'https://api.openai.com/v1/chat/completions'
        api_key = getattr(config, 'OPENAI_KEY', None)
        if api_key is None or api_key.strip() == "YOUR KEY":
            raise ValueError("OPENAI_KEY 未在 config.py 中设置。")
        default_model = 'gpt-3.5-turbo'
    else:
        raise ValueError(f"未知 provider: {_provider}. 目前仅支持 'openai' 与 'deepseek'.")

    # 2. 组装请求 payload
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": model or default_model,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias
    }

    # 3. 重试机制
    retries = 0
    while True:
        try:
            r = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=timeout
            )
            if r.status_code != 200:
                retries += 1
                time.sleep(1)
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(1)
            retries += 1

    r = r.json()
    return [choice['message']['content'] for choice in r['choices']]


def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": "text-davinci-003",
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True
    }
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=10
            )  
            if r.status_code != 200:
                time.sleep(2)
                retries += 1
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
    r = r.json()
    return r['choices']


