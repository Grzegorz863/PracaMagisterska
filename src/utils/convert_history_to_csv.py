import csv
import pickle

with open('F:\\PracaMagisterska\\saved_models\\history\\model_88.history', 'rb') as file:
    history = pickle.load(file)

    with open("F:\\PracaMagisterska\\saved_models\\history\\model_88_history.csv", "w") as outfile:
        writer = csv.writer(outfile, delimiter=";")
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))


