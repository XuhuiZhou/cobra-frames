from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

# inputs = tokenizer("Given a statement or a conversational snippet,         as well as its situational context, speaker, and listener identity,         generate the rating of those scenarios. \n\n[Statement] You’re so much less man-hating than I expected a [women’s college] lesbian to be![/]\n[Speaker identity/characteristics] white lesbian[/]\n[Plausibility of the speaker] yes[/]\n\n[Statement] Rap music and its artists are whats holding african-america back[/]\n[Speaker identity/characteristics] black person[/]\n[Plausibility of the speaker] yes[/]\n\n[Statement] why are you friends with her? she is retarded![/]\n[Speaker identity/characteristics] student with learning disability[/]\n[Plausibility of the speaker] no[/]\n\n\
# [Statement] so i don't call black people niggers[/]\n[Speaker identity/characteristics] black person[/]\n[Plausibility of the speaker]", return_tensors="pt")
inputs = tokenizer(
    "Given a statement or a conversational snippet,         as well as its situational context, speaker, and listener identity,         generate the rating of those scenarios. \n\n[Statement] You’re so much less man-hating than I expected a [women’s college] lesbian to be![/]\n[Speaker identity/characteristics] white lesbian[/]\n[Plausibility of the speaker] This is somewhat plausible. Even though the statement indicates that the speaker misunderstood college lesbians, the speaker can still be a lesbian, so the answer yes.[/]\n\n[Statement] why are you friends with her? she is retarded![/]\n[Speaker identity/characteristics] student with learning disability[/]\n[Plausibility of the speaker] This is implausible because someone with learning disability is unlikely to call other people retarded, so the answer is no.[/]\n\n[Statement] Rap music and its artists are whats holding african-america back[/]\n[Speaker identity/characteristics] black person[/]\n[Plausibility of the speaker] This is plausible because the speaker can be black even if the statement is discussing African-American people, biased statements can also come from in-group, so the answer is yes.[/]\n\n\
[Statement] so i don't call black people niggers[/]\n[Speaker identity/characteristics] black person[/]\n[Plausibility of the speaker]",
    return_tensors="pt",
)
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
