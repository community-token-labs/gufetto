// Included by http_server_test.cpp after its transport helpers.
// Fixtures are fake-backend route recordings, NOT live-client/model evidence.

std::string ClientFixture(std::string_view name) {
  const auto path =
      std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" /
      "client-recipes" / name;
  std::ifstream file(path, std::ios::binary);
  assert(file);
  return {std::istreambuf_iterator<char>(file), {}};
}

struct ClientSse {
  std::vector<gufo::json::Value> events;
  bool done{false};
};

ClientSse ReplayClientSse(std::string_view wire, std::size_t fragment) {
  ClientSse result;
  std::string pending;
  // Replay byte-sized network fragments as well as complete recorded frames.
  for (std::size_t offset = 0; offset < wire.size(); offset += fragment) {
    pending.append(wire.substr(offset, fragment));
    for (;;) {
      const auto end = pending.find("\n\n");
      if (end == std::string::npos)
        break;
      const auto frame = pending.substr(0, end);
      pending.erase(0, end + 2);
      const auto data = frame.find("data: ");
      if (data == std::string::npos)
        continue;  // SSE comments/keepalive.
      assert(!result.done);
      const auto payload = frame.substr(data + 6);
      if (payload == "[DONE]") {
        result.done = true;
      } else {
        auto event = gufo::json::parse(payload);
        if (event.find("type")) {
          // Typed APIs dispatch by the SSE event name, not just JSON data.
          assert(frame.starts_with("event: "));
          assert(frame.substr(7, frame.find('\n') - 7) ==
                 event.member_str("type"));
        }
        result.events.push_back(std::move(event));
      }
    }
  }
  assert(pending.empty());
  return result;
}

gufo::json::Value NormalizeClientEvent(const gufo::json::Value& value) {
  using gufo::json::Value;
  if (value.is_object()) {
    auto normalized = Value::object();
    for (const auto& [key, item] : value.members()) {
      // Only generated identifiers and creation timestamps are volatile.
      if (key == "id" || key == "call_id" || key == "item_id")
        normalized[key] = "<generated-id>";
      else if (key == "created" || key == "created_at")
        normalized[key] = 0;
      else
        normalized[key] = NormalizeClientEvent(item);
    }
    return normalized;
  }
  if (value.is_array()) {
    auto normalized = Value::array();
    for (const auto& item : value.items())
      normalized.push_back(NormalizeClientEvent(item));
    return normalized;
  }
  return value;
}

void CompareClientRecording(const ClientSse& recorded, const ClientSse& live) {
  assert(recorded.done == live.done);
  assert(recorded.events.size() == live.events.size());
  for (std::size_t i = 0; i < recorded.events.size(); ++i)
    assert(NormalizeClientEvent(recorded.events[i]).dump() ==
           NormalizeClientEvent(live.events[i]).dump());
}

std::size_t RequiredClientSize(const gufo::json::Value& object,
                               const std::string& key) {
  const auto* value = object.find(key);
  assert(value && value->is_number());
  const auto result = value->as_size();
  assert(value->as_double() == static_cast<double>(result));
  return result;
}

struct ClientCall {
  std::string id, name, arguments;
};

void CheckClientUsage(std::string_view protocol,
                      const gufo::json::Value& usage) {
  if (protocol == "chat") {
    assert(usage.member_size("prompt_tokens") == 10);
    assert(usage.member_size("completion_tokens") == 1);
    assert(usage.find("prompt_tokens_details")->member_size("cached_tokens") ==
           8);
  } else if (protocol == "responses") {
    assert(usage.member_size("input_tokens") == 10);
    assert(usage.member_size("output_tokens") == 1);
    assert(usage.find("input_tokens_details")->member_size("cached_tokens") ==
           8);
  } else {
    assert(usage.member_size("input_tokens") == 2);
    assert(usage.member_size("output_tokens") == 1);
    assert(usage.member_size("cache_read_input_tokens") == 8);
    assert(RequiredClientSize(usage, "cache_creation_input_tokens") == 0);
  }
}

ClientCall CheckClientToolStream(std::string_view protocol,
                                 const ClientSse& sse) {
  ClientCall call;
  bool terminal = false;
  bool usage = false;
  std::string complete_arguments;
  std::size_t next_sequence = 0;
  std::set<std::size_t> blocks;
  // This bounded recording contains exactly one tool block/call.
  const std::vector<std::string> expected =
      protocol == "responses"
          ? std::vector<std::string>{"response.created",
                                     "response.in_progress",
                                     "response.output_item.added",
                                     "response.function_call_arguments.delta",
                                     "response.function_call_arguments.done",
                                     "response.output_item.done",
                                     "response.completed"}
          : std::vector<std::string>{
                "message_start",       "content_block_start",
                "content_block_delta", "content_block_stop",
                "message_delta",       "message_stop"};
  if (protocol != "chat") {
    assert(sse.events.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
      assert(sse.events[i].member_str("type") == expected[i]);
  }
  for (const auto& event : sse.events) {
    if (protocol == "chat") {
      if (const auto* choices = event.find("choices")) {
        for (const auto& choice : choices->items()) {
          if (choice.member_str("finish_reason") == "tool_calls")
            terminal = true;
          if (const auto* delta = choice.find("delta")) {
            assert(delta->member_str("content").empty());
            if (const auto* calls = delta->find("tool_calls")) {
              for (const auto& part : calls->items()) {
                assert(RequiredClientSize(part, "index") == 0);
                if (!part.member_str("id").empty()) {
                  assert(call.id.empty());
                  call.id = part.member_str("id");
                  assert(part.member_str("type") == "function");
                }
                if (const auto* function = part.find("function")) {
                  call.name += function->member_str("name");
                  call.arguments += function->member_str("arguments");
                }
              }
            }
          }
        }
        if (choices->size() == 0 && event.find("usage")) {
          assert(terminal);
          CheckClientUsage(protocol, *event.find("usage"));
          usage = true;
        }
      }
    } else if (protocol == "responses") {
      assert(RequiredClientSize(event, "sequence_number") == next_sequence++);
      const auto type = event.member_str("type");
      if (type == "response.output_item.added") {
        const auto& item = *event.find("item");
        if (item.member_str("type") == "function_call") {
          assert(call.id.empty());
          call.id = item.member_str("call_id");
          call.name = item.member_str("name");
        }
      } else if (type == "response.function_call_arguments.delta") {
        call.arguments += event.member_str("delta");
      } else if (type == "response.function_call_arguments.done") {
        complete_arguments = event.member_str("arguments");
      } else if (type == "response.completed") {
        const auto& response = *event.find("response");
        assert(response.member_str("status") == "completed");
        usage = response.find("usage") != nullptr;
        assert(usage);
        CheckClientUsage(protocol, *response.find("usage"));
        terminal = true;
        std::size_t count = 0;
        for (const auto& item : response.find("output")->items()) {
          if (item.member_str("type") == "function_call") {
            ++count;
            assert(item.member_str("call_id") == call.id);
            assert(item.member_str("arguments") == call.arguments);
          }
        }
        assert(count == 1);
      }
    } else {
      const auto type = event.member_str("type");
      if (type == "content_block_start") {
        assert(blocks.insert(RequiredClientSize(event, "index")).second);
        const auto& block = *event.find("content_block");
        if (block.member_str("type") == "tool_use") {
          assert(call.id.empty());
          call.id = block.member_str("id");
          call.name = block.member_str("name");
        }
      } else if (type == "content_block_delta") {
        assert(blocks.contains(RequiredClientSize(event, "index")));
        const auto& delta = *event.find("delta");
        if (delta.member_str("type") == "input_json_delta")
          call.arguments += delta.member_str("partial_json");
      } else if (type == "content_block_stop") {
        assert(blocks.erase(RequiredClientSize(event, "index")) == 1);
      } else if (type == "message_delta") {
        assert(blocks.empty());
        assert(event.find("delta")->member_str("stop_reason") == "tool_use");
        usage = event.find("usage") != nullptr;
        assert(usage);
        CheckClientUsage(protocol, *event.find("usage"));
      } else if (type == "message_stop") {
        assert(usage && blocks.empty());
        terminal = true;
      }
    }
  }
  assert(terminal && usage);
  assert(sse.done == (protocol == "chat"));
  assert(!call.id.empty() && call.name == "read_file");
  const auto arguments = gufo::json::parse(call.arguments);
  assert(arguments.member_str("path") == "note.txt");
  if (protocol == "responses")
    assert(complete_arguments == call.arguments);
  return call;
}

void CheckClientAnswerStream(std::string_view protocol, const ClientSse& sse) {
  std::string text;
  bool terminal = false, usage = false;
  for (const auto& event : sse.events) {
    if (protocol == "chat") {
      for (const auto& choice : event.find("choices")->items()) {
        terminal |= choice.member_str("finish_reason") == "stop";
        if (const auto* delta = choice.find("delta")) {
          assert(!delta->find("tool_calls"));
          text += delta->member_str("content");
        }
      }
      if (const auto* value = event.find("usage"); value && !value->is_null()) {
        CheckClientUsage(protocol, *value);
        usage = true;
      }
    } else if (protocol == "responses") {
      if (event.member_str("type") == "response.output_text.delta")
        text += event.member_str("delta");
      if (event.member_str("type") == "response.completed") {
        terminal = true;
        CheckClientUsage(protocol, *event.find("response")->find("usage"));
        usage = true;
      }
    } else {
      if (event.member_str("type") == "content_block_delta")
        text += event.find("delta")->member_str("text");
      if (event.member_str("type") == "message_delta") {
        assert(event.find("delta")->member_str("stop_reason") == "end_turn");
        CheckClientUsage(protocol, *event.find("usage"));
        usage = true;
      }
      terminal |= event.member_str("type") == "message_stop";
    }
  }
  assert(text == "Read complete." && terminal && usage);
  assert(sse.done == (protocol == "chat"));
}

void TestClientRecipeToolLoops() {
  using gufo::json::Value;
  for (const std::string protocol : {"chat", "responses", "messages"}) {
    const auto fixture = gufo::json::parse(ClientFixture(protocol + ".json"));
    const auto endpoint = fixture.member_str("endpoint");
    // Both stored SSE replay and the current real HTTP route must satisfy the
    // same client-facing assembly contract, independent of generated IDs.
    for (const auto size : {1U, 7U, 65536U})
      CheckClientToolStream(
          protocol, ReplayClientSse(ClientFixture(protocol + ".sse"), size));
    RunningServer server;
    server.backend->SetOutput(fixture.member_str("backend_output"));
    const auto request = *fixture.find("request");
    const auto first = server.Post(endpoint, request.dump());
    ExpectStatus(first, 200);
    assert(first.find("text/event-stream") != std::string::npos);
    const auto wire = ChunkedBody(first);
    const auto live = ReplayClientSse(wire, 1);
    CompareClientRecording(ReplayClientSse(ClientFixture(protocol + ".sse"), 7),
                           live);
    const auto call = CheckClientToolStream(protocol, live);
    auto next = request;
    if (protocol == "chat") {
      auto assistant = gufo::json::parse(
          R"({"role":"assistant","content":null,"tool_calls":[]})");
      auto tool = Value::object();
      tool["id"] = call.id;
      tool["type"] = "function";
      tool["function"]["name"] = call.name;
      tool["function"]["arguments"] = call.arguments;
      assistant["tool_calls"].push_back(tool);
      next["messages"].push_back(assistant);
      auto result =
          gufo::json::parse(R"({"role":"tool","content":"fixture contents"})");
      result["tool_call_id"] = call.id;
      next["messages"].push_back(result);
    } else if (protocol == "responses") {
      auto tool = Value::object();
      tool["type"] = "function_call";
      tool["call_id"] = call.id;
      tool["name"] = call.name;
      tool["arguments"] = call.arguments;
      next["input"].push_back(tool);
      auto result = gufo::json::parse(
          R"({"type":"function_call_output","output":"fixture contents"})");
      result["call_id"] = call.id;
      next["input"].push_back(result);
    } else {
      auto tool = Value::object();
      tool["type"] = "tool_use";
      tool["id"] = call.id;
      tool["name"] = call.name;
      tool["input"] = gufo::json::parse(call.arguments);
      auto assistant =
          gufo::json::parse(R"({"role":"assistant","content":[]})");
      assistant["content"].push_back(tool);
      next["messages"].push_back(assistant);
      auto result = gufo::json::parse(
          R"({"type":"tool_result","content":"fixture contents"})");
      result["tool_use_id"] = call.id;
      auto user = gufo::json::parse(R"({"role":"user","content":[]})");
      user["content"].push_back(result);
      next["messages"].push_back(user);
    }
    server.backend->SetOutput("Read complete.");
    const auto second = server.Post(endpoint, next.dump());
    ExpectStatus(second, 200);
    const auto replay = server.backend->LastCall().chat;
    assert(replay.messages.size() == 3);
    const auto& assistant = replay.messages[1];
    assert(assistant.tool_calls.size() == 1);
    assert(assistant.tool_calls[0].id == call.id);
    assert(assistant.tool_calls[0].name == call.name);
    assert(assistant.tool_calls[0].arguments.size() == 1);
    assert(assistant.tool_calls[0].arguments[0].name == "path");
    assert(assistant.tool_calls[0].arguments[0].value == "note.txt");
    const auto& result = replay.messages[2];
    assert(result.role == gufo::tokenization::ChatRole::kTool);
    assert(result.tool_call_id == call.id &&
           result.content == "fixture contents");
    CheckClientAnswerStream(protocol, ReplayClientSse(ChunkedBody(second), 7));
  }
}
