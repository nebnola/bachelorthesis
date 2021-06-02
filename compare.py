"""Load different fits and write their error into a csv to compare"""
import csv
import fit

with open('fitcomparison.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["method", "dim", "error"])
    for dim in range(3, 7):
        files = {'exp-pos': f"bestexpfit{dim}.json",
                 'exp': f"oldfit{dim}.json",
                 'general-ou': f"bestoufit{dim}.json",
                 'general-ou-init': f"oubasedfit{dim}.json"}
        for name, file in files.items():
            fitres = fit.FitResult.from_file(file)
            writer.writerow([name, dim, fitres.max_dev()])
