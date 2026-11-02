#!/usr/bin/env python3
"""Compile verbatim JSON helpers from src/main.cpp and verify RFC 8259 edge cases."""
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MAIN_CPP = ROOT / 'src/main.cpp'


def extract_json_block(source: str) -> str:
    start_marker = '// RFC 8259 JSON object helpers'
    end_marker = '// Single synthesis helper'
    start = source.index(start_marker)
    end = source.index(end_marker)
    return source[start:end]


HARNESS = r"""
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

__JSON_BLOCK__

int main() {
    // 1. Unicode BMP escapes (\u00e9, \u4f60\u597d)
    std::string u1 = json_get_str(R"({"text":"Caf\u00e9 \u4f60\u597d"})", "text");
    assert(u1 == "Café 你好");

    // 2. UTF-16 surrogate pair (\ud83d\ude80 -> U+1F680 🚀)
    std::string u2 = json_get_str(R"({"text":"Launch \ud83d\ude80"})", "text");
    assert(u2 == "Launch 🚀");

    // 3. Key name appearing inside earlier string value
    std::string k1 = json_get_str(R"({"note":"text","junk":"wrong","text":"correct"})", "text");
    assert(k1 == "correct");

    // 4. Escaped quote + key-value pattern inside earlier string value
    std::string k2 = json_get_str(R"({"note":"\"text\":\"wrong\"","text":"right"})", "text");
    assert(k2 == "right");

    // 5. Nested object/array containing shadowed key
    std::string k3 = json_get_str(R"({"meta":{"text":"nested"},"arr":[{"text":"in_arr"}],"text":"top"})", "text");
    assert(k3 == "top");

    // 6. Numeric fields and key-in-string isolation
    float temp = json_get_float(R"({"note":"\"temperature\": 0.1","temperature":0.75})", "temperature", -1.0f);
    assert(std::fabs(temp - 0.75f) < 1e-6f);

    // 7. Strict integer validation for vocoder_chunk_frames
    int32_t v = 0;
    bool present = false;
    assert(json_get_strict_int(R"({"text":"hi","vocoder_chunk_frames":200})", "vocoder_chunk_frames", v, present));
    assert(present && v == 200);
    assert(!json_get_strict_int(R"({"text":"hi","vocoder_chunk_frames":1.5})", "vocoder_chunk_frames", v, present));
    assert(!json_get_strict_int(R"({"text":"hi","vocoder_chunk_frames":"200"})", "vocoder_chunk_frames", v, present));
    assert(!json_get_strict_int(R"({"text":"hi","vocoder_chunk_frames":999999999999999})", "vocoder_chunk_frames", v, present));

    // 8. Reject malformed JSON / unpaired surrogates / unescaped control chars
    JsonFieldValue val;
    bool found = false, valid = false;
    assert(!json_find_top_level_field(R"({"text":"unterminated})", "text", val, found, &valid) && !valid);
    assert(!json_find_top_level_field(R"({"text":"\ud83d"})", "text", val, found, &valid) && !valid);
    assert(!json_find_top_level_field(R"({"text":"\ude80"})", "text", val, found, &valid) && !valid);
    assert(!json_find_top_level_field("{\"text\":\"raw\nnewline\"}", "text", val, found, &valid) && !valid);
    assert(!json_find_top_level_field(R"({"text":"ok"} trailing)", "text", val, found, &valid) && !valid);

    // 9. json_escape control characters
    std::string esc = json_escape(std::string("a\"b\\c\nd\x01", 8));
    assert(esc == "a\\\"b\\\\c\\nd\\u0001");

    printf("{\"tests\":9,\"passed\":9,\"unicode_actual\":\"%s\",\"key_in_value_actual\":\"%s\",\"surrogate_ok\":true}\n",
           u1.c_str(), k1.c_str());
    return 0;
}
"""


def main():
    source = MAIN_CPP.read_text(encoding='utf-8')
    block = extract_json_block(source)
    cpp_code = HARNESS.replace('__JSON_BLOCK__', block)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cpp_file = tmp_path / 'probe.cpp'
        bin_file = tmp_path / 'probe'
        cpp_file.write_text(cpp_code, encoding='utf-8')
        subprocess.run(['clang++', '-std=c++17', '-O2', str(cpp_file), '-o', str(bin_file)], check=True)
        run = subprocess.run([str(bin_file)], capture_output=True, text=True, check=True)
        report = json.loads(run.stdout)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
