class AnthropicClient(AnthropicBaseClient):
    _system_message: str
    _messages: List[dict[str, Any]]

    def __init__(self, system_message: str, model: str = ANTHROPIC_BASE_MODEL):
        super().__init__()
        self._system_message = system_message
        self._messages = []
        self._model = model

    def _add_message(self, message, _type: Literal["text", "jpeg", "png"] = "text", _image: str = None, _role: Literal["user", "assistant"] = "user"):
        content = []
        if _type == "text":
            content.append({"type": "text", "text": message})
        else:
            if message:
                content.append({"type": "text", "text": message})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": f"image/{_type}", "data": _image}
            })
        message_dict = {
            "role": _role,
            "content": content
        }
        self._messages.append(message_dict)

    def get_messages(self):
        return self._messages

    async def call_llm(self, user_query, _type: Literal["text", "jpeg", "png"] = "text", _image: str = None) -> str:
        self._ensure_client()
        self._add_message(user_query, _type, _image)
        messages = self.get_messages()
        def _sync_call():
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=1000,
                temperature=0.8,
                system=self._system_message,
                messages=messages
            )
            if resp.content and len(resp.content) > 0:
                return resp.content[0].text
            return ""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _sync_call)
        try:
            result = result.decode("utf-8")
        except Exception:
            pass
        return str(result)
