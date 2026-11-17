get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

if(NOT EXISTS "${DEST}")
    message(STATUS "Downloading ${NAME} from ggml-org/models...")
endif()

# CA_BUNDLE (optional, from tests/CMakeLists.txt): a CA file for TLS verification. A portable cmake (mingw)
# ships no CA store, so without it the download fails with status 60 and every fixture test is Not Run.
set(_tls_ca_args)
if(DEFINED CA_BUNDLE AND NOT "${CA_BUNDLE}" STREQUAL "" AND EXISTS "${CA_BUNDLE}")
    set(_tls_ca_args TLS_CAINFO "${CA_BUNDLE}")
endif()

file(DOWNLOAD
    "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true"
    "${DEST}"
    TLS_VERIFY ON
    ${_tls_ca_args}
    EXPECTED_HASH ${HASH}
    STATUS status
)

list(GET status 0 code)

if(NOT code EQUAL 0)
    list(GET status 1 msg)
    # A failed download leaves a 0-byte DEST behind; remove it so the next run does not start from a stub.
    file(REMOVE "${DEST}")
    if("${_tls_ca_args}" STREQUAL "")
        message(FATAL_ERROR "Failed to download ${NAME}: ${msg} (no CA bundle was passed; set LLAMA_TEST_CA_BUNDLE "
                            "or SSL_CERT_FILE at configure time, or install certifi in the Python cmake found)")
    endif()
    message(FATAL_ERROR "Failed to download ${NAME}: ${msg} (CA bundle: ${CA_BUNDLE})")
endif()
