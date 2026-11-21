"""One command: train everything, evaluate, write results. Usage: python run_all.py   (SMOKE=1 for a tiny offline test)"""
import train_baselines, train_transformers, evaluate

if __name__ == "__main__":
    train_baselines.main()
    train_transformers.main()
    evaluate.main()
