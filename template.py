TEMPLATES_cot = [
    '\n\nDeliberate thoughtfully before submitting your selection.',
    '\n\nTake your time to consider your answer carefully before responding.',
    "\n\nLets' think step by step",
    '\n\nCarefully think about your response before replying.',
    "\n\nBefore you respond, please ensure you've thought about your answer thoroughly.",
    "\n\nPlease consider your answer thoughtfully before responding.",
    "\n\nBefore answering, take a moment to carefully consider your response.",
    "\n\nTake a deliberate moment to think through your answer before responding.",
    "\n\nThink carefully before you return the answer.",
    "\n\nTake a moment to reflect on your answer prior to responding."
]

TEMPLATES = {
    0: """Choose the most appropriate instruction among the [INSTRUCTIONS] that instructs the generation of the given [RESPONSE].
You must choose one of the following four options when answering.
Generate your answer without any explanation.
Think carefully before you return the answer.
{shot}
[RESPONSE] is as follows:
{condition}
[INSTRUCTIONS] are as follows
{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}{cot}""",
    1: """Please proceed by selecting the instruction from the list provided below that most accurately directs the creation of the specified [RESPONSE]. Your selection process should adhere to the following guidelines:
Review the instructions listed under [INSTRUCTIONS] thoroughly to determine which one aligns best with the [RESPONSE] provided.
From the options presented, make a singular choice without furnishing any accompanying rationale.
Exercise deliberate consideration prior to finalizing your response.{shot}
The [RESPONSE] to be referenced is delineated as follows:
{condition}

Presented [INSTRUCTIONS] for your consideration include:

{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}{cot}""",
    2: """Select the instruction from the list below that best explains how to produce the given response. 
Please select only one option and provide your answer directly, without any explanation. {shot}
Here's the response you need to match with the correct instruction:

Response:
{condition}

Instructions to choose from:

{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}{cot}""",
    3: """Begin by examining the instructions listed below, identifying the one that most precisely guides you in crafting the desired [RESPONSE]. Follow these steps carefully:

Carefully read through the [INSTRUCTIONS] listed to find the one that best matches the [RESPONSE] outlined.
Select only one option from the list, without providing any reasons for your choice.
Take your time to think over your decision before settling on it.{shot}
The [RESPONSE] in question is defined as follows:
{condition}
The [INSTRUCTIONS] available for selection are:

{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}{cot}""",
    4: """Kindly follow these steps to choose the appropriate directive for the given [RESPONSE] from the options below. Ensure your decision-making process adheres to these protocols:

Carefully evaluate each of the listed [INSTRUCTIONS] to identify the one that best matches the [RESPONSE] described.
Select only one option from those provided, without providing an explanation for your choice.
Take your time to thoughtfully consider your selection before confirming it.
{shot}
The options [INSTRUCTIONS] available for selection are:

{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}

The [RESPONSE] in question is defined as follows:
{condition}{cot}""",
    5: """Identify the instruction from [INSTRUCTIONS] that accurately facilitates the generation of the targeted [RESPONSE].
Ensure to pick only one option from the given four as your response.
Submit your choice plainly, without any explanation.
Reflect deeply prior to delivering your final answer.
{shot}
The [INSTRUCTIONS] provided for your choice are:

{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}

Here is the [RESPONSE] for consideration:
{condition}{cot}""",
    6: """Please carefully follow the instructions below to select the most suitable directive for the specified [RESPONSE] from the available choices. It's important that your selection process is in line with these guidelines:

Examine all provided [INSTRUCTIONS] thoroughly to determine which one accurately aligns with the [RESPONSE] provided.
Choose only one of the given options, and there's no need to justify your selection.
Deliberately consider your choice before finalizing it.{shot}
The available [INSTRUCTIONS] options to choose from are:

{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}

The [RESPONSE] to be matched is outlined as follows:
{condition}{cot}""",
    7: """Please carefully follow the instructions below to select:
Examine all provided [INSTRUCTIONS] thoroughly to determine which one accurately aligns with the [RESPONSE] provided.
Choose only one of the given [INSTRUCTIONS].
Deliberately consider your choice before finalizing it.{shot}
[INSTRUCTIONS]

{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}

[RESPONSE]
{condition}{cot}""",
    8: """You must identify the correct instruction that produces the specified [RESPONSE]:
{shot}
Begin by understanding the described [RESPONSE] as stated:
{condition}

Review this list of [INSTRUCTIONS] to determine which matches the [RESPONSE]:

{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}
Select the instruction that most accurately aligns with the [RESPONSE] mentioned.{cot}""",
    9: """The following is a result generated through an LLM.

{condition}

You must guess which Instruction the above result derived from.{shot}
Choose one of the following four options.
{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}{cot}""",
    10: """You must choose one of the following four options:
{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}

Which instruction derives the following statement? Think carefully before you response:
{condition}{shot}{cot}""",
    11: """Select the most appropriate option that guides the creation of the mentioned [RESPONSE], after examining the following list of [INSTRUCTIONS] to find the one that corresponds:
{shot}
{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}
Ensure you are familiar with the specified [RESPONSE], detailed as:
{condition}{cot}""",
    12: """I want you to select the correct instruction for producing the specified [RESPONSE], {shot}follow these steps:

    Understand the [RESPONSE] described below:
    {condition}

    Review this set of [INSTRUCTIONS] to identify the one that matches the [RESPONSE]:

    {option_mark} {option1}
    {option_mark} {option2}
    {option_mark} {option3}
    {option_mark} {option4}
Select the instruction that most accurately reflects the guidance for creating the mentioned [RESPONSE].{cot}""",
    13: """Select the most appropriate option that guides the creation of the described [RESPONSE]:{shot}
[RESPONSE]: 
{condition}

Choose one of the following [INSTRUCTIONS]:
[INSTRUCTIONS]:

{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}{cot}""",
    14: """Identify the instruction among the following [INSTRUCTIONS] that corresponds with the [RESPONSE]:{shot}

{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}
After familiarizing yourself with the outlined [RESPONSE] as specified:
{condition}
Select the most appropriate option that accurately guides the creation of the mentioned [RESPONSE].{cot}""",
    15: """Your task is to select the instruction from the list provided below that most accurately directs the creation of the specified [RESPONSE].{shot}
 
Inspect the following instruction options:
{option_mark} {option1}
{option_mark} {option2}
{option_mark} {option3}
{option_mark} {option4}

The [RESPONSE] to be referenced is delineated as follows:
{condition}

From the options presented, make only a single choice.{cot}""",
}