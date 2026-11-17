// audio-capture.h -- microphone capture into an audio_ring, through the miniaudio already vendored in this tree
// (vendor/miniaudio, 0.11.x, public domain / MIT-0). whisper.cpp's live example uses SDL2 for this
// (examples/common-sdl.cpp, read at 6e4ab854); miniaudio gives the same callback-into-a-ring shape without a new
// third-party dependency, and its "null" backend is a real capture device with no hardware behind it, which is
// what the tests open.
#pragma once

#include "audio-stream.h"

#include <string>
#include <vector>

struct audio_capture;

// backend: "" = miniaudio's default order for this OS (WASAPI/DirectSound/WinMM on Windows, ...), "null" = the null
// device (produces silence; for tests). device_index < 0 = the default capture device.
// The device delivers mono float32 at `sample_rate` (miniaudio converts from the hardware format).
audio_capture * audio_capture_open(const std::string & backend, int device_index, int sample_rate, audio_ring & ring,
                                   std::string & err);
bool            audio_capture_start(audio_capture * c, std::string & err);
void            audio_capture_stop(audio_capture * c);
void            audio_capture_close(audio_capture * c);
std::string     audio_capture_device_name(const audio_capture * c);
std::string     audio_capture_backend_name(const audio_capture * c);

// names of the capture devices the backend reports, in index order
bool audio_capture_list(const std::string & backend, std::vector<std::string> & names, std::string & err);
