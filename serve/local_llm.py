"""Local query understanding: Qwen2.5-1.5B-Instruct with constrained decoding.

Same job as the Anthropic path in query_understanding.py, run on weights we host
ourselves — no API key, no per-query cost, no third-party dependency in the
request path.

The piece that makes this viable is **constrained decoding**. A hosted API
guarantees schema-valid JSON server-side (`strict: true`). Running weights
yourself there is no such guarantee, and a 1.5B model asked politely for JSON
will sometimes emit prose, trailing commas, or a missing field. `outlines` masks
the token distribution at each step so only tokens keeping the output
schema-valid can be sampled — the JSON is structurally guaranteed by
construction rather than by parsing and retrying.

Loaded lazily: importing this pulls ~3GB into memory, which the rule-based and
Anthropic paths have no need for.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# outlines needs a self-contained JSON Schema. This mirrors FILTER_TOOL's schema
# but spells nullable fields as anyOf rather than a type array, which is the
# form the schema-to-automaton compiler handles consistently.
def build_schema(countries: list[str], varieties: list[str]) -> dict:
    """Schema with country/variety as enums over the real column values.

    Constraining these matters more than it looks. Declared as free-form
    strings, a 1.5B model emits values that don't exist in the corpus -- one run
    produced `"United States林业"` as a country. Listing the valid values in the
    prompt does nothing to prevent that; only the schema does. As enums, the
    automaton makes every invalid value structurally unreachable, so the model
    can be wrong about *which* country but never about whether it is one.
    """
    schema = json.loads(json.dumps(FILTER_SCHEMA))  # deep copy
    schema["properties"]["country"] = {
        "anyOf": [{"type": "string", "enum": countries}, {"type": "null"}]
    }
    schema["properties"]["variety"] = {
        "anyOf": [{"type": "string", "enum": varieties}, {"type": "null"}]
    }
    return schema


FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "country": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "variety": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "color": {
            "anyOf": [
                {"type": "string", "enum": ["red", "white", "rose", "sparkling",
                                            "dessert", "fortified", "orange"]},
                {"type": "null"},
            ]
        },
        "price_min": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "price_max": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "points_min": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "points_max": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "query": {"type": "string"},
    },
    "required": ["country", "variety", "color", "price_min", "price_max",
                 "points_min", "points_max", "query"],
    "additionalProperties": False,
}

_model = None
_tokenizer = None
_generators: dict[str, object] = {}


def _build() -> tuple:
    """Load the model once. Uses CUDA when present (ZeroGPU), else MPS, else CPU."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import outlines
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # .to(device) rather than device_map: device_map requires `accelerate`, and
    # an eager .to("cuda") at load time is the pattern ZeroGPU expects -- it
    # intercepts the call, packs the weights, and streams them into VRAM on the
    # first @spaces.GPU entry.
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
    _model = outlines.from_transformers(hf_model, _tokenizer)
    return _model, _tokenizer


def _generator_for(countries: list[str], varieties: list[str]):
    """Compiling a schema to an automaton is expensive, so cache per value set."""
    import outlines

    model, _ = _build()
    key = f"{len(countries)}:{len(varieties)}"
    if key not in _generators:
        schema = build_schema(countries, varieties)
        _generators[key] = outlines.Generator(model, outlines.json_schema(json.dumps(schema)))
    return _generators[key]


def build_prompt(raw_query: str, countries: list[str], varieties: list[str]) -> str:
    """Render the extraction request through the model's chat template."""
    _, tokenizer = _build()
    system = (
        "Extract wine search filters from the request. "
        "Use an exact country name from KNOWN COUNTRIES and an exact grape from "
        "KNOWN VARIETIES, or null if the request does not imply one. "
        "Set color when implied, including by a named grape. "
        "Use null for anything not mentioned. "
        "'query' is the descriptive part of the request.\n\n"
        f"KNOWN COUNTRIES: {', '.join(countries)}\n\n"
        f"KNOWN VARIETIES: {', '.join(varieties)}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": raw_query}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract(raw_query: str, countries: list[str], varieties: list[str],
            max_new_tokens: int = 200) -> dict:
    """Extract filters locally. The result is schema-valid by construction."""
    generator = _generator_for(countries, varieties)
    prompt = build_prompt(raw_query, countries, varieties)
    return json.loads(generator(prompt, max_new_tokens=max_new_tokens))
