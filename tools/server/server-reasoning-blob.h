#pragma once

// /v1/responses reasoning replay through `encrypted_content`.
// Row S69-STRATA-S2-RESPONSES-API-PARITY-CUSTOM-NAMESPACE-TOOLS-INCOMPLETE-FAILED-USAGE-2026-10-07-01.
//
// WHY: a stateless Responses client (store=false) asks for include=["reasoning.encrypted_content"] and, on the next turn,
// sends back each reasoning item with that blob and often WITHOUT its readable content. This server used to emit an
// empty blob, so such a client replayed nothing and the model lost its own earlier reasoning; an item with no readable
// text was skipped (server-chat.cpp).
//
// WHAT: when (and only when) the request includes "reasoning.encrypted_content", the reasoning item's encrypted_content
// is PREFIX + base64(reasoning text). On input, an item whose content has no text but whose encrypted_content carries
// PREFIX is decoded and replayed as reasoning_content. A blob from another server (no PREFIX, or not decodable) is
// still skipped, exactly as before. Without the include, the output keeps encrypted_content "" as before.
//
// NOT ENCRYPTED: the blob is opaque by contract (clients must not parse it), not confidential. This is a local server and
// the same text is already sent in the item's readable content, so hiding it would protect nothing. Anyone may hand the
// server any reasoning text through the readable content field already, so accepting it back here adds no new input.

#include "base64.hpp"

#include <string>

namespace server_reasoning_blob {

inline const std::string & prefix() {
    static const std::string p = "ikfork-rs1:";
    return p;
}

inline const char * include_key() {
    return "reasoning.encrypted_content";
}

inline std::string encode(const std::string & text) {
    return prefix() + base64::encode(text);
}

// true and the text when the blob is one of ours and decodes; false for a foreign or damaged blob
inline bool decode(const std::string & blob, std::string & text) {
    const std::string & p = prefix();
    if (blob.size() <= p.size() || blob.compare(0, p.size(), p) != 0) {
        return false;
    }
    try {
        text = base64::decode(blob.substr(p.size()), base64::alphabet::standard);
    } catch (const std::exception &) {
        return false;
    }
    return !text.empty();
}

} // namespace server_reasoning_blob
