from langchain_openai import ChatOpenAI
import httpx

from promptflow.core import tool
from promptflow.connections import CustomStrongTypeConnection
from promptflow.contracts.types import PromptTemplate
from promptflow.contracts.types import Secret
from promptflow.tools.common import render_jinja_template
from promptflow.tools.exception import InvalidConnectionType




class OCPConnection(CustomStrongTypeConnection):
    """OCP type connection.

    :param token: The token from the OCP AI Model Server.
    :type token: Secret
    :param endpoint: The api base.
    :type endpoint: String
    """
    token: Secret
    endpoint: str = "This is a fake api base."


@tool(streaming_option_parameter="stream")
def ocp_llm(connection: OCPConnection,
            prompt: PromptTemplate,
            model: str = "",
            temperature: float = 0.1,
            max_tokens: int = None,
            **kwargs,) -> str:

    rendered_prompt = render_jinja_template(prompt, trim_blocks=True, keep_trailing_newline=True, **kwargs)

    if isinstance(connection, OCPConnection):
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=2,
            api_key=f"{connection.secrets['token']}",
            base_url=f"{connection.endpoint}/v1",
            http_client=httpx.Client(verify=False)
        )
        ai_msg = llm.invoke(rendered_prompt)
        return ai_msg.content
    else:
        error_message = f"Not Support connection type '{type(connection).__name__}' for OCP llm. " \
                        "Connection type should be in OCPConnection."
        raise InvalidConnectionType(message=error_message)
