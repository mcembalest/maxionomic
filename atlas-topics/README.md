# NomiClaude Atlas Topics improvement

We are trying to improve a Nomic Atlas topic model with Claude prompts. 

The current approach one-shots this, prompting claude with a printout of the current Nomic keyword tree, and with a tool call / structured output / json schema (these are all jargon for pretty much the same thing these days) we get a new topic model from Claude with new topic names.

## Instructions

### Login

First login with your Nomic API KEY:

```bash
nomic login nk-...
```

If you need to test this in a different environment, make sure to use that environment during your login (e.g. `nomic login staging nk-...`)

### Run

```bash
python main.py "your-dataset-name"
```

e.g.

```bash
python main.py "ai-policy-recommendations"
```

```bash
python main.py "airline-reviews-data"
```

```bash
python main.py "100k-sample-common-vulnerabilities-and-exposures-cves"
```

## Todo

- set it up for uv usage

- datasets to use: wine, mcdonalds

- explore how big can an atlas topic model be for this method? Currently we are at ~600/700 topics in claude input