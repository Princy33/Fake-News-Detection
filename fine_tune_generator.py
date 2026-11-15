import pandas as pd
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# ----------------------------------
# 1. Load Dataset
# ----------------------------------

fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

# Use titles as headline training data
fake_titles = fake_df["title"].dropna()
true_titles = true_df["title"].dropna()

texts = list(fake_titles) + list(true_titles)

dataset = Dataset.from_dict({"text": texts})

# ----------------------------------
# 2. Load Tokenizer & Model
# ----------------------------------

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# ----------------------------------
# 3. Tokenization
# ----------------------------------

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ----------------------------------
# 4. Data Collator
# ----------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ----------------------------------
# 5. Training Arguments
# ----------------------------------

training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100
)

# ----------------------------------
# 6. Trainer
# ----------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# ----------------------------------
# 7. Train Model
# ----------------------------------

trainer.train()

# ----------------------------------
# 8. Save Model
# ----------------------------------

model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

print("✅ Fine-tuning complete!")