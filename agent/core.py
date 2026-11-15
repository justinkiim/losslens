"""Minimal tool-calling agent loop (Anthropic Messages API content-block format).

Design goals: small, auditable, and safe by default.
  * Tools declare a JSON schema; the model chooses which to call.
  * Tool exceptions are returned to the model as error results (never crash the loop).
  * A hard step budget stops runaway loops.
  * Every model turn and tool call is appended to a JSONL audit log.
"""
import json
import time


class Tool(object):
    def __init__(self, name, description, input_schema, fn):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.fn = fn

    def schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class Agent(object):
    def __init__(self, client, tools, system="", max_steps=8, audit_path=None):
        self.client = client
        self.tools = dict((t.name, t) for t in tools)
        self.system = system
        self.max_steps = max_steps
        self.audit_path = audit_path
        self.trace = []  # list of dicts: step, tool, input, ok

    def _audit(self, record):
        record = dict(record)
        record["ts"] = time.time()
        self.trace.append(record)
        if self.audit_path:
            with open(self.audit_path, "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

    def _run_tool(self, step, block):
        name = block.get("name")
        args = block.get("input") or {}
        tool = self.tools.get(name)
        if tool is None:
            self._audit({"step": step, "tool": name, "input": args, "ok": False, "error": "unknown tool"})
            return {"type": "tool_result", "tool_use_id": block["id"],
                    "content": "Error: unknown tool %r" % name, "is_error": True}
        try:
            result = tool.fn(**args)
            text = result if isinstance(result, str) else json.dumps(result, default=str)
            self._audit({"step": step, "tool": name, "input": args, "ok": True})
            return {"type": "tool_result", "tool_use_id": block["id"], "content": text}
        except Exception as exc:  # returned to the model so it can self-correct
            self._audit({"step": step, "tool": name, "input": args, "ok": False, "error": str(exc)})
            return {"type": "tool_result", "tool_use_id": block["id"],
                    "content": "Error: %s" % exc, "is_error": True}

    def run(self, user_message):
        messages = [{"role": "user", "content": user_message}]
        schemas = [t.schema() for t in self.tools.values()]
        for step in range(1, self.max_steps + 1):
            reply = self.client.create(self.system, messages, schemas)
            messages.append({"role": "assistant", "content": reply["content"]})
            if reply.get("stop_reason") != "tool_use":
                text = "".join(b.get("text", "") for b in reply["content"] if b.get("type") == "text")
                self._audit({"step": step, "final": True})
                return {"text": text, "steps": step, "stopped": "end_turn", "messages": messages}
            results = [self._run_tool(step, b) for b in reply["content"] if b.get("type") == "tool_use"]
            messages.append({"role": "user", "content": results})
        self._audit({"step": self.max_steps, "final": False, "error": "max_steps reached"})
        return {"text": "", "steps": self.max_steps, "stopped": "max_steps", "messages": messages}
