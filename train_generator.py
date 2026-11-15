from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="./fine_tuned_gpt2",
    tokenizer="./fine_tuned_gpt2"
)

def generate_headline(topic, tone):

    if tone == "Sensational":
        prompt = f"Breaking shocking news about {topic}:"
    else:
        prompt = f"News update about {topic}:"

    result = generator(
        prompt,
        max_length=25,
        temperature=0.9,
        do_sample=True,
        num_return_sequences=1
    )

    return result[0]["generated_text"]