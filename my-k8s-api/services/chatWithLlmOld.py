def chat_with_llm(user_message: str, history: list | None = None) -> Dict[str, Any]:
    history = prune_history(history or [])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    # Handle CSV requests (unchanged logic)
    csv_keywords = ["download csv", "export csv"]
    if any(keyword in user_message.lower() for keyword in csv_keywords):
        inferred_params = infer_parameters_from_history(history, user_message)
        if not inferred_params["namespace"] or not inferred_params["pod"]:
            missing = []
            if not inferred_params["namespace"]:
                missing.append("namespace")
            if not inferred_params["pod"]:
                missing.append("pod")
            return {
                "assistant": f"Please specify the {', '.join(missing)} for the CSV download.",
                "history": history
            }
        result = generate_csv_link(
            namespace=inferred_params["namespace"],
            pod=inferred_params["pod"],
            range=inferred_params["range"]
        )
        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": "inferred_csv_call"
        })
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_DEF,
            tool_choice="none",
            max_tokens=800,
        )
        final_msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": final_msg.content})
        front_history = [m for m in messages if m["role"] != "system"]
        return {"assistant": final_msg.content, "history": front_history}

    # First LLM call to decide tool usage
    from openai import BadRequestError
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_DEF,
            tool_choice="auto",
            max_tokens=800,
        )
    except BadRequestError as err:
        human = (
            "⚠️ I couldn't execute that request (invalid tool arguments). "
            "Try specifying: namespace, metric (cpu/memory), duration like 30m/2h/1d."
        )
        return {"assistant": human, "history": history or []}
    
    assistant_msg = resp.choices[0].message
    tool_calls = getattr(assistant_msg, "tool_calls", None) or []
    messages.append({
        "role": "assistant",
        "content": assistant_msg.content,
        "tool_calls": tool_calls
    })

    # Execute tool calls with enhanced error handling
    for tool_call in tool_calls:
        fn = FUNC_MAP.get(tool_call.function.name)
        if not fn:
            continue
        try:
            args = json.loads(tool_call.function.arguments)
            result = fn(**args)
        except TypeError as e:
            error_msg = str(e)
            match = re.search(r"missing (\d) required positional argument: '(\w+)'", error_msg)
            if match:
                arg_name = match.group(2)
                assistant_response = f"Error: Missing required argument '{arg_name}' for {tool_call.function.name}. Please provide it."
            else:
                assistant_response = f"Error executing {tool_call.function.name}: {error_msg}"
            return {"assistant": assistant_response, "history": history}
        except Exception as e:
            assistant_response = f"Error executing {tool_call.function.name}: {str(e)}"
            return {"assistant": assistant_response, "history": history}
        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })

    # Second LLM call for final response
    if tool_calls:
        resp2 = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_DEF,
            tool_choice="none",
            max_tokens=800,
        )
        final_msg = resp2.choices[0].message
        messages.append({"role": "assistant", "content": final_msg.content})

    front_history = [m for m in messages if m["role"] != "system"]
    last_msg = messages[-1]
    assistant_content = last_msg.get("content") or "[Error: No response generated from the assistant.]"
    
    return {"assistant": assistant_content, "history": front_history}