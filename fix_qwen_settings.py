#!/usr/bin/env python3
import json, sys
p = "/Users/takasurazeem/.qwen/settings.json"
d = json.load(open(p))
pcfg = d["modelProviders"]["openai"][0]["generationConfig"]
pcfg["streamIdleTimeoutMs"] = 3600000
pcfg["timeout"] = 43200000
if d.get("model") and "name" in d["model"]:
    d["model"].pop("name")
json.dump(d, open(p, "w"), indent=2)
print("updated ok")
print(json.dumps(pcfg, indent=2))
print("model:", json.dumps(d.get("model")))