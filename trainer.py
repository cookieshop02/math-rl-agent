from trl import GRPOConfig, GRPOTrainer
from reward import batch_reward_fn


def train(model, tokenizer, dataset, output_dir="./output"):
    print("\nStarting GRPO Training...")
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=128,
        beta=0.04,
        learning_rate=1e-5,
        warmup_steps=5,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        use_cpu=True,
        remove_unused_columns=False,
        seed=42,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=batch_reward_fn,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done! Loss: {result.training_loss:.4f} | Saved to: {output_dir}")
    return trainer