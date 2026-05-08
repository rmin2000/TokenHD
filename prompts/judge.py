math_process_error_prompt = '''You are given a question and its corresponding solution. Your task is to analyze whether the solution completely and correctly solves the given problem.

**Task**:
1. Carefully examine the solution step by step
2. Identify if there are any errors in the reasoning, logic, facts, calculations, or final result
3. If errors are found, extract the exact erroneous parts from the solution and list them using error tags

**Instructions**:
- Analyze the solution thoroughly for factual errors, mathematical errors, logical flaws, incorrect reasoning or wrong final answers
- If the solution contains any errors, extract the exact text from the solution that contains the error
- **IMPORTANT**: Use the original text from the solution without any modifications - maintain the exact same format, wording, punctuation, and mathematical notation as in the original
- Keep the extracted error text **complete** and **relevant** to the specific error. Do **NOT** extract only mathematical symbols or formulas.
- Tag each error using `<error n></error n>` format, starting from index 1 and numbering sequentially
- If the solution is completely correct, simply state "No errors!"

**Input Format**:
```
Problem: [PROBLEM_STATEMENT]

Solution: [SOLUTION_TEXT]
```

**Output Format**:
If errors are found:
```
<error 1>[EXACT_TEXT_FROM_SOLUTION_1]</error 1>
<error 2>[EXACT_TEXT_FROM_SOLUTION_2]</error 2>
...
```

If no errors are found:
```
No errors!
```

Please analyze the following:

Problem: {problem}

Solution: {solution}'''


code_process_error_prompt = '''You are given a coding problem and its corresponding solution. Your task is to analyze whether the solution completely and correctly solves the given problem.

**Your Task**:

1. Carefully examine the solution step by step, including the code implementation and any accompanying explanation.
2. Identify if there are any errors in the syntax, logic, algorithm, implementation, or final result.
3. If errors are found, extract the exact erroneous parts from the solution and list them using error tags.

**Instructions**:

- **You MUST identify all distinct errors in the solution. Do not stop after finding the first error.**
- Treat as an error any of the following:
  - syntax or runtime errors,
  - logical flaws or incorrect algorithms,
  - wrong or incomplete handling of edge cases,
  - violations of the problem's constraints,
  - obviously incorrect or incomplete implementation of the required logic,
  - misuse or misunderstanding of built-in functions or data types.
- **Use only original text from the solution**: do not modify wording, formatting, punctuation, or indentation.
- Tag each error using `<error n></error n>` format, starting from 1 and numbering sequentially.
- If the solution is completely correct, output exactly: `No errors!`

**Input Format**:

```
Problem: [PROBLEM_STATEMENT]

Solution: [SOLUTION_TEXT]
```

**Output Format**:

If errors are found:
```
<error 1>[EXACT_TEXT_FROM_SOLUTION_1]</error 1>
<error 2>[EXACT_TEXT_FROM_SOLUTION_2]</error 2>
...
```

If no errors:
```
No errors!
```

Please analyze the following:

Problem: {problem}

Solution: {solution}'''


sci_process_error_prompt = '''You are given a science/engineering problem and its corresponding solution. Your task is to analyze whether the solution completely and correctly solves the given problem.

**Task**:
1. Carefully examine the solution step by step
2. Identify if there are any errors in the reasoning, physics/chemistry principles, calculations, or final result
3. If errors are found, extract the exact erroneous parts from the solution and list them using error tags

**Instructions**:
- Analyze the solution thoroughly for factual errors, scientific errors, logical flaws, or incorrect reasoning
- If the solution contains any errors, extract the exact text from the solution that contains the error
- **IMPORTANT**: Use the original text from the solution without any modifications
- Tag each error using `<error n></error n>` format, starting from index 1 and numbering sequentially
- If the solution is completely correct, simply state "No errors!"

**Input Format**:
```
Problem: [PROBLEM_STATEMENT]

Solution: [SOLUTION_TEXT]
```

**Output Format**:
If errors are found:
```
<error 1>[EXACT_TEXT_FROM_SOLUTION_1]</error 1>
...
```

If no errors are found:
```
No errors!
```

Please analyze the following:

Problem: {problem}

Solution: {solution}'''
