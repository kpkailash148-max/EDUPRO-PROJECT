from src.modeling import evaluate
if __name__ == "__main__":
    metrics, data = evaluate()
    print("Training complete. Model comparison:")
    print(metrics.to_string(index=False))
    print(f"\nRows: {len(data)}; outputs saved in outputs/.")
