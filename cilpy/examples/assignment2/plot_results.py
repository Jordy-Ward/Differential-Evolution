import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Define the directory where the CSV files are stored
    out_dir = "out"
    if not os.path.exists(out_dir):
        print(f"Could not find the '{out_dir}' directory. Please run the script from the assignment2 directory.")
        return

    # Define the list of problems and solvers based on the experiment
    problems = ["Rastrigin", "Ackley", "Griewank", "Schwefel", "Michalewicz", "Levy", "Salomon", "Alpine1"]
    solvers = ["DE_rand_2_bin", "DE_rand_2_AX", "DE_rand_2_UNDX", "DE_rand_2_PCX", "DE_rand_2_SPX"]

    # Create a directory to save the generated plots
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    for problem in problems:
        plt.figure(figsize=(10, 6))
        plt.title(f"Average Convergence on {problem} (30 Runs)", fontsize=14)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Average Best Fitness (Accuracy)", fontsize=12)
        
        has_data = False
        
        for solver in solvers:
            csv_path = os.path.join(out_dir, f"{problem}_{solver}.out.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Group by iteration and compute the mean accuracy across all 30 runs
                avg_accuracy = df.groupby("iteration")["accuracy"].mean()
                plt.plot(avg_accuracy.index, avg_accuracy.values, label=solver, linewidth=2)
                has_data = True
            else:
                print(f"Warning: Data for {problem} with {solver} not found at {csv_path}")

        if has_data:
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            # You can uncomment the line below to use a logarithmic scale if the fitness drops very quickly
            # plt.yscale('log') 
            
            output_path = os.path.join(plots_dir, f"{problem}_convergence.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_path}")
        
        # Close the figure to free up memory for the next plot
        plt.close()

if __name__ == "__main__":
    main()
