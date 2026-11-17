#pragma once

// --chat-template-kwargs-force: chat-template kwargs the server pins. --chat-template-kwargs is only a default (a request's
// chat_template_kwargs replaces it); a key set here replaces whatever the request sent, so an operator can hold a served
// model at e.g. enable_thinking=false whatever the client asks. Design only from the default/force split of a
// per-model template-variable config (the force set wins over the request); no code copied.

#include <map>
#include <string>

// kwargs: key -> JSON value text. Every key of force replaces the same key in kwargs (or is added).
static inline void server_template_kwargs_apply_force(
        std::map<std::string, std::string> & kwargs,
        const std::map<std::string, std::string> & force) {
    for (const auto & kv : force) {
        kwargs[kv.first] = kv.second;
    }
}
