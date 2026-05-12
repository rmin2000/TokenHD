"""Prompt for restoring annotated spans to verbatim substrings of the original text."""

RESTORE_PROMPT = """\
You are given one original text and multiple extracted text segments that were supposedly extracted from the original text, but may contain slight variations, errors, or modifications.
1. **Original Text**: A long document or passage
2. **Extracted Text**: A segment that was supposedly extracted from the original text, but may contain slight variations, errors, or modifications

**Task**: For each extracted text segment, locate the corresponding section in the original text that best matches it, and output the exact original text segment with appropriate tags.

**Instructions**:
- The extracted text may have minor differences from the original, such as:
  - Spelling variations or typos
  - Punctuation differences
  - Minor word substitutions
  - Formatting changes
- Find the most similar section in the original text for each extracted segment. If multiple potential matches exist for one segment, choose the one with the highest similarity.
- Only return the portion of the original text that corresponds to the extracted segment. Do NOT add extra content before or after the matching section, and do NOT remove any content that should be included.
- **IMPORTANT**: Return the **EXACT** text from the original document that corresponds to each extracted segment. Pay attention to differences, such as *whitespace*, *punctuation*, *brackets*, and *line breaks*, etc.
- If no reasonable match is found, output "NO_MATCH_FOUND".

**Input Format**:
```
Original Text: [ORIGINAL_TEXT]

<extract1>[EXTRACTED_TEXT_1]</extract1>
<extract2>[EXTRACTED_TEXT_2]</extract2>
<extract3>[EXTRACTED_TEXT_3]</extract3>
...
```

**Output Format**:
```
<result1>[EXACT_SEGMENT_FROM_ORIGINAL_1]</result1>
<result2>[EXACT_SEGMENT_FROM_ORIGINAL_2]</result2>
<result3>[EXACT_SEGMENT_FROM_ORIGINAL_3]</result3>
...
```

Please process the following input:

Original Text: {original_text}

Extracted Text: {extracted_text}"""
